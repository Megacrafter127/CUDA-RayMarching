/*
 * simpledraw.c
 *
 *  Created on: 16.07.2019
 *      Author: Megacrafter127
 */

#include "simpledraw.h"

#include <SDL2/SDL.h>
#include <assert.h>
#include <time.h>
#include <math.h>

static void __attribute__((constructor)) construct() {
	if(SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr,"InitError: %s\nExiting\n",SDL_GetError());
		exit(1);
	}
}

static void __attribute__((destructor)) destruct() {
	SDL_Quit();
}

static inline SDL_Surface *extractSurface(SDL_Window *win) {
	if(!win) return NULL;
	SDL_Surface *screen = SDL_GetWindowSurface(win);
	if(!screen) {
		SDL_DestroyWindow(win);
		return NULL;
	}
	screen->userdata=win;
	assert(!SDL_MUSTLOCK(screen));
	return screen;
}

SDL_Surface *createSurface(unsigned width, unsigned height, const char *title) {
	return extractSurface(SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS));
}

SDL_Surface *createFullscreenSurface(const char *title, int grabInput, int real) {
	SDL_DisplayMode dm;
	if(SDL_GetCurrentDisplayMode(0,&dm)) {
		return NULL;
	}
	return extractSurface(SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dm.w, dm.h, SDL_WINDOW_SHOWN | (real?SDL_WINDOW_FULLSCREEN:SDL_WINDOW_FULLSCREEN_DESKTOP) | (grabInput?SDL_WINDOW_INPUT_GRABBED:0)));
}

int renderSurface(SDL_Surface *surface) {
	return SDL_UpdateWindowSurface(surface->userdata);
}

void destroySurface(SDL_Surface *surface) {
	SDL_DestroyWindow(surface->userdata);
}

int defaultEventFunction(SDL_Event *event, void *data) {
	return event->type == SDL_QUIT;
}

#define SECONDS_PER_CLOCK (1.0f/CLOCKS_PER_SEC)
static struct timespec totime(float seconds) {
	if(seconds<0) seconds=0;
	struct timespec ret={tv_sec:seconds,tv_nsec:fmodf(seconds,1.0f)*1000000000L};
	return ret;
}

int autoDraw(SDL_Surface *surface, drawFunc drawFunction, float deltaT, eventFunc eventFunction, void *data) {
	if(!eventFunction) eventFunction=defaultEventFunction;
	int c=0;
	clock_t start=clock();
	for(size_t i=0;!(c=drawFunction(surface,i,data));i++,start=clock()) {
		c=renderSurface(surface);
		if(c) return c;
		SDL_Event event;
		struct timespec remTime=totime(deltaT-(SECONDS_PER_CLOCK*(clock()-start)));
		do{
			while(SDL_PollEvent(&event)) {
				c=eventFunction(&event,data);
				if(c) return c;
			}
		} while(nanosleep(&remTime,&remTime));
	}
	return c;
}

typedef struct {
	pixelFunc func;
	void *data;
} pixelFuncContainer;

static int drawPixelsFunc(SDL_Surface *surface, size_t frame, void *data) {
	pixelFuncContainer *pfc=data;
	for(size_t y=0;y<surface->h;y++) for(size_t x=0;x<surface->w;x++) {
		int r=pfc->func(surface->pixels+y*surface->pitch+x*surface->format->BytesPerPixel,
			surface->format, x, y, frame, pfc->data);
		if(r) return r;
	}
	return 0;
}

int autoDrawPixels(SDL_Surface *surface,pixelFunc pixelFunction,float deltaT,eventFunc eventFunction,void *data) {
	pixelFuncContainer pfc={func:pixelFunction,data:data};
	return autoDraw(surface,drawPixelsFunc,deltaT,eventFunction,&pfc);
}

