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

SDL_Surface *createSurface(unsigned width, unsigned height, const char *title) {
	SDL_Window *win = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS);
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

int renderSurface(SDL_Surface *surface) {
	return SDL_UpdateWindowSurface(surface->userdata);
}

void destroySurface(SDL_Surface *surface) {
	SDL_DestroyWindow(surface->userdata);
}

int defaultEventFunction(SDL_Event *event, void *data) {
	return event->type == SDL_QUIT;
}

int autoDraw(SDL_Surface *surface, drawFunc drawFunction, float deltaT, eventFunc eventFunction, void *data) {
	if(!eventFunction) eventFunction=defaultEventFunction;
	const struct timespec waitTime={tv_sec:(long)deltaT,tv_nsec:(long)(fmodf(deltaT,1.0f)*1000000000L)};
	int c=0;
	for(size_t i=0;!(c=drawFunction(surface,i,data));i++) {
		c=renderSurface(surface);
		if(c) return c;
		SDL_Event event;
		struct timespec remTime=waitTime;
		do{
			while(SDL_PollEvent(&event)) {
				c=eventFunction(&event,data);
				if(c) return c;
			}
		} while(nanosleep(&remTime,&remTime));
	}
	return c;
}
