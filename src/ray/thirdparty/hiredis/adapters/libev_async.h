/*
 * Copyright (c) 2010-2011, Pieter Noordhuis <pcnoordhuis at gmail dot com>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __HIREDIS_LIBEV_ASYNC_H__
#define __HIREDIS_LIBEV_ASYNC_H__
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <ev.h>
#include "../hiredis.h"
#include "../async.h"

typedef struct redisLibevEvents {
    redisAsyncContext *context;
    struct ev_loop *loop;
    int reading, writing;
    ev_async rev, wev;
} redisLibevEvents;

static void redisLibevReadEvent(EV_P_ ev_async *watcher, int revents) {
#if EV_MULTIPLICITY
    ((void)loop);
#endif
    ((void)revents);
    fprintf(stderr, "libev read event happen.\n");
    redisLibevEvents *e = (redisLibevEvents*)watcher->data;
    redisAsyncHandleRead(e->context);
}

static void redisLibevWriteEvent(EV_P_ ev_async *watcher, int revents) {
#if EV_MULTIPLICITY
    ((void)loop);
    fprintf(stderr, "ev multiplicity\n");
#endif
    ((void)revents);
    
    fprintf(stderr, "libev write event happen.\n");
    redisLibevEvents *e = (redisLibevEvents*)watcher->data;
    redisAsyncHandleWrite(e->context);
}

static void redisLibevAddRead(void *privdata) {
    fprintf(stderr, "libev add read\n");
    redisLibevEvents *e = (redisLibevEvents*)privdata;
    struct ev_loop *loop = e->loop;
    ((void)loop);
    if (!e->reading) {
        e->reading = 1;
		fprintf(stderr, "libev trigger read\n");
        ev_async_send(EV_A_ &e->rev);
    } else {
        fprintf(stderr, "in reading, do nothing\n");
    }
}

static void redisLibevDelRead(void *privdata) {
    redisLibevEvents *e = (redisLibevEvents*)privdata;
    struct ev_loop *loop = e->loop;
    ((void)loop);
    if (e->reading) {
        e->reading = 0;
        fprintf(stderr, "libev stop read\n");
        //ev_io_stop(EV_A_ &e->rev);
    }
}

static void redisLibevAddWrite(void *privdata) {
    fprintf(stderr, "libev add write\n");
    redisLibevEvents *e = (redisLibevEvents*)privdata;
    struct ev_loop *loop = e->loop;
    ((void)loop);
    if (!e->writing) {
        e->writing = 1;
	    fprintf(stderr, "libev trigger write\n");
        ev_async_send(EV_A_ &e->wev);
    } else {
        fprintf(stderr, "in writing, do nothing\n");
    }
}

static void redisLibevDelWrite(void *privdata) {
    redisLibevEvents *e = (redisLibevEvents*)privdata;
    struct ev_loop *loop = e->loop;
    ((void)loop);
    if (e->writing) {
        e->writing = 0;
        fprintf(stderr, "libev stop write\n");
        //ev_io_stop(EV_A_ &e->wev);
    }
}

static void redisLibevCleanup(void *privdata) {
    redisLibevEvents *e = (redisLibevEvents*)privdata;
    redisLibevDelRead(privdata);
    redisLibevDelWrite(privdata);
    free(e);
}

static int redisLibevAttach(EV_P_ redisAsyncContext *ac) {
    //redisContext *c = &(ac->c);
    redisLibevEvents *e;

    /* Nothing should be attached when something is already attached */
    if (ac->ev.data != NULL)
        return REDIS_ERR;

    /* Create container for context and r/w events */
    e = (redisLibevEvents*)malloc(sizeof(*e));
    e->context = ac;
#if EV_MULTIPLICITY
    //fprintf(stderr, "ev multiplicity\n");
    e->loop = loop;
#else
    e->loop = NULL;
#endif
    e->reading = e->writing = 0;
    e->rev.data = e;
    e->wev.data = e;

    /* Register functions to start/stop listening for events */
    ac->ev.addRead = redisLibevAddRead;
    ac->ev.delRead = redisLibevDelRead;
    ac->ev.addWrite = redisLibevAddWrite;
    ac->ev.delWrite = redisLibevDelWrite;
    ac->ev.cleanup = redisLibevCleanup;
    ac->ev.data = e;

    /* Initialize read/write events */
    ev_async_init(&e->rev,redisLibevReadEvent);
    ev_async_init(&e->wev,redisLibevWriteEvent);
    ev_async_start(loop, &e->rev);
    ev_async_start(loop, &e->wev);
    return REDIS_OK;
}

#endif
