""" Word2Vec example, with the majority taken from
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import random
from tempfile import gettempdir
import zipfile
import pickle

import time
from datetime import datetime
import subprocess
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import ray

ray.init(num_cpus=400)



# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# pylint: disable=redefined-outer-name
def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                       local_filename)
    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


#filename = maybe_download('text8.zip', 31344016)
filename = 'text8.zip'

# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 7051


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# Step 4: Build and train a skip-gram model.

batch_size = 10
embedding_size = 10000  # Dimension of the embedding vector.
skip_window = 4  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 5  # Number of negative examples to sample.
print("Embedding size:", embedding_size)

@ray.remote(num_cpus=20)
class Word2VecModel(object):
    def __init__(self):
        # Input data.
        with tf.name_scope('inputs'):
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Ops and variables pinned to the CPU because
        # of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(tf.random_uniform(
                                         [vocabulary_size, embedding_size],
                                         -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(tf.truncated_normal(
                                      [vocabulary_size, embedding_size],
                                      stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative
        # labels each time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.nce_loss(
                                       weights=nce_weights,
                                       biases=nce_biases,
                                       labels=self.train_labels,
                                       inputs=embed,
                                       num_sampled=num_sampled,
                                       num_classes=vocabulary_size))


        # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.GradientDescentOptimizer(0.024).minimize(self.loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        # Add variable initializer.
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
        self.sess.run(init)

    def step(self, weights, inputs, labels):
        feed_dict = {self.train_inputs: inputs, self.train_labels: labels}
        self.variables.set_weights(weights)
        self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return self.variables.get_weights()

    def get_weights(self):
        return self.variables.get_weights()

    def final_embeddings(self):
        return normalized_embeddings.eval()
    
num_steps = 10
actor_num = 10


actor_list = [Word2VecModel.remote() for i in range(actor_num)]
weights = ray.get(actor_list[0].get_weights.remote())

total_net_time = 0
total_calc_time = 0
#start_time = datetime.now()
for iteration in range(num_steps):
    start_time = datetime.now()
    weights_ids = ray.put(weights)
    end_time = datetime.now()
    total_net_time += (end_time - start_time).total_seconds()
    new_weights_ids = []
    batch_inputs_list = []
    batch_labels_list = []
    start_time = datetime.now()
    for actor in actor_list:
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        batch_inputs_list.append(batch_inputs)
        batch_labels_list.append(batch_labels)
   
    #start_time = datetime.now()
    for i in range(actor_num):
        new_weight_id = actor.step.remote(weights_ids, batch_inputs_list[i], batch_labels_list[i])
        new_weights_ids.append(new_weight_id)
    
    ready, _ = ray.wait(new_weights_ids, num_returns=len(new_weights_ids), timeout = 5000)
    #print(ready)
    end_time = datetime.now()
    total_calc_time += (end_time - start_time).total_seconds()
    
    start_time = datetime.now()
    new_weights_list = ray.get(new_weights_ids)
    end_time = datetime.now()
    total_net_time += (end_time - start_time).total_seconds() 
    print((end_time - start_time).total_seconds())
    '''
    cnt = 0
    for weight_dict in new_weights_list:
        for variable in weight_dict:
            cnt += len(weight_dict[variable])
    print("CNT:", cnt)
    break
    '''
    weights = {variable: sum(weight_dict[variable] for weight_dict in new_weights_list) / actor_num for variable in new_weights_list[0]}
    if iteration % 100 == 0:
        print("Iteration {}: weights are {}".format(iteration, weights))
    byteOutput = subprocess.check_output(["df", "-h"])
    print(byteOutput.decode('UTF-8').rstrip())
    if iteration == 9:
        pickle.dump(new_weights_list, open("weight.obj", 'w'))
        byteOutput = subprocess.check_output(["ls", "-atrl"])
        print(byteOutput.decode('UTF-8').rstrip())

rate = float(total_calc_time)/float(total_net_time)
print("rate:", rate)
#print("Total time:", (datetime.now()-start_time).total_seconds)
#print("Avg calc time:", float(total_calc_time)/float(num_steps))
#print("Avg pull time:", float(total_pull_time)/float(num_steps))
