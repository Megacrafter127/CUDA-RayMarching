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
SDL_Surface *createSurface(unsigned width, unsigned height, const char *title);

/**
 * Ensures that any changes made to the surface are reflected in the associated window.
 * @param surface	the surface created with createSurface
 * @return			0 on success, a non-0 value if an error occurs
 */
int renderSurface(SDL_Surface *surface);

/**
 * Frees up all resources associated with the surface
 * @param surface	the surface created with createSurface
 */
void destroySurface(SDL_Surface *surface);

/**
 * A function to draw the given frame on the SDL_Surface
 * @param surface	the surface to draw on. It is guaranteed to be locked if needed.
 * @param frame		the frame counter
 * @param data		user supplied data
 * @return			0 on success, non-0 if an error occurs or this should be the last rendered frame
 */
typedef int (*drawFunc)(SDL_Surface *surface, size_t frame, void *data),
/**
 * A function to process SDL_Events
 * @param event		the event to process
 * @param data		user supplied data
 * @return			0 on success, non-0 if an error occurs or no more frames should be rendered
 */
		(*eventFunc)(SDL_Event *event, void *data);

/**
 * The default event processing function.
 * Returns 1 if event->type is SDL_Quit, otherwise returns 0.
 * Has no side effects.
 * @param event		the event to process
 * @param data		user supplied data
 * @return			0 on success, non-0 if an error occurs or no more frames should be rendered
 */
int defaultEventFunction(SDL_Event *event, void* data);

/**
 * Automatic drawing loop.
 * @param surface			the SDL_Surface to draw on
 * @param drawFunction		the drawFunc used to draw each frame
 * @param deltaT			minimum amount of time between frames in seconds
 * @param eventFunction		the eventFunc used to process events generated during the loop
 * @param data				user supplied data to be passed to drawFunction and eventFunction
 * @return					the first non-0 return of either drawFunction or eventFunction
 */
int autoDraw(SDL_Surface *surface, drawFunc drawFunction, float deltaT, eventFunc eventFunction, void *data);

#endif /* SIMPLEDRAW_H_ */
