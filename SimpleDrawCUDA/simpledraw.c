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
#include <stdio.h>

static void __attribute__((constructor)) construct() {
	if(SDL_Init(SDL_INIT_VIDEO)) {
		fprintf(stderr,"InitError: %s\nExiting\n",SDL_GetError());
		exit(1);
	}
}

static void __attribute__((destructor)) destruct() {
	SDL_Quit();
}

const static char GL_CONTEXT_NAME[] = "glctx";

static inline SDL_Window *extractSurface(SDL_Window *win) {
	if(!win) return win;
	SDL_GLContext ctx = SDL_GL_CreateContext(win);
	SDL_SetWindowData(win,GL_CONTEXT_NAME,ctx);

	if(SDL_GL_SetSwapInterval(-1)) {
		fprintf(stderr,"Unable to use adaptive vsync: %s\nFalling back to normal vsync.\n",SDL_GetError());
		SDL_GL_SetSwapInterval(1);
	}

	return win;
}

SDL_Window *createWindow(unsigned width, unsigned height, const char *title) {
	return extractSurface(SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_BORDERLESS));
}

SDL_Window *createFullscreenWindow(const char *title, int grabInput, int real) {
	SDL_DisplayMode dm;
	if(SDL_GetCurrentDisplayMode(0,&dm)) {
		return NULL;
	}
	return extractSurface(SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, dm.w, dm.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | (real?SDL_WINDOW_FULLSCREEN:SDL_WINDOW_FULLSCREEN_DESKTOP) | (grabInput?SDL_WINDOW_INPUT_GRABBED:0)));
}

void destroyWindow(SDL_Window *win) {
	SDL_GLContext ctx=SDL_GetWindowData(win,GL_CONTEXT_NAME);
	SDL_GL_DeleteContext(ctx);
	SDL_DestroyWindow(win);
}
