/*
 * simpledraw.h
 *
 *  Created on: 16.07.2019
 *      Author: Megacrafter127
 */
#ifndef SIMPLEDRAW_H_
#define SIMPLEDRAW_H_

#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_events.h>

/**
 * Creates a window with an SDL_Surface to draw on.
 * @param width		the width the surface should have  
 * @param height	the height the surface should have
 * @return			a pointer to the created surface, or NULL if an error occurred
 */
SDL_Window *createWindow(unsigned width, unsigned height, const char *title);

SDL_Window *createFullscreenWindow(const char *title, int grabInput, int real);

/**
 * Frees up all resources associated with the surface
 * @param surface	the surface created with createSurface
 */
void destroyWindow(SDL_Window *surface);

/**
 * A function to process SDL_Events
 * @param event		the event to process
 * @param data		user supplied data
 * @return			0 on success, non-0 if an error occurs or no more frames should be rendered
 */
typedef int (*eventFunc)(SDL_Event *event, void *data);

/**
 * The default event processing function.
 * Returns 1 if event->type is SDL_Quit, otherwise returns 0.
 * Has no side effects.
 * @param event		the event to process
 * @param data		user supplied data
 * @return			0 on success, non-0 if an error occurs or no more frames should be rendered
 */
int defaultEventFunction(SDL_Event *event, void* data);

typedef union {
	struct{
		float r,g,b,a;
	};
	float raw[4];
} color_t;

#endif /* SIMPLEDRAW_H_ */
