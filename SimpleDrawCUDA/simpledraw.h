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

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a window with an SDL_Surface to draw on.
 * @param width		the width the surface should have  
 * @param height	the height the surface should have
 * @return			a pointer to the created surface, or NULL if an error occurred
 */
extern SDL_Window *createWindow(unsigned width, unsigned height, const char *title);

extern SDL_Window *createFullscreenWindow(const char *title, int grabInput, int real);

/**
 * Frees up all resources associated with the surface
 * @param surface	the surface created with createSurface
 */
extern void destroyWindow(SDL_Window *surface);

/**
 *
 * @param surface
 * @param bounds
 * @param frame
 * @param userData
 */
typedef void (*launchDrawKernel)(cudaSurfaceObject_t surface, dim3 bounds, size_t frame, const void *userData, cudaStream_t stream);

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
extern int defaultEventFunction(SDL_Event *event, void* data);

/**
 * Host function to be called before the draw kernel launches.
 * To facilitate returning as soon as possible, inter-frame data changes should be performed post kernel finish.
 * @param frame			frame counter
 * @param data			user supplied data, unmodifiable to discourage unnecessary runtime of this function
 * @param dyn_shared	how much dynamically allocated shared memory the draw kernel should have. Initialized to 0.
 * @return				0 on success, non-0 otherwise
 */
typedef int (*preFrameFunc)(size_t frame, const void *data, cudaStream_t stream),
/**
 * Host function to be called after the draw kernel finished.
 * Used for inter-frame changes to user data.
 *
 * The potentially longer runtime can be afforded by running this function while the device is copying the frame back to host memory.
 * @param frame		frame counter
 * @param data		user supplied data
 * @return			0 on success, non-0 otherwise or if this is the last frame to be rendered.
 */
			(*postFrameFunc)(size_t frame, void *data, cudaStream_t stream);

/**
 * Default preFrame function.
 * Does nothing but return 0.
 * @param frame			frame counter
 * @param data			user supplied data, unmodifiable to discourage unnecessary runtime of this function
 * @param dyn_shared	how much dynamically allocated shared memory the draw kernel should have. Initialized to 0.
 * @return				0 on success, non-0 otherwise
 */
extern int defaultPreFrame(size_t frame, const void *data, cudaStream_t stream);
/**
 * Default postFrame function.
 * Does nothing but return 0.
 * @param frame		frame counter
 * @param data		user supplied data
 * @return			0 on success, non-0 otherwise or if this is the last frame to be rendered.
 */
extern int defaultPostFrame(size_t frame, void *data, cudaStream_t stream);

/**
 * Draw loop that automatically runs the drawKernel each frame.
 * The drawKernel will run the supplied cudaFunc once for each pixel on the surface.
 * @param surface			The surface to draw on
 * @param func				The CUDA function to draw each pixel
 * @param center			If set to 0, position will be passed as is to func. If set to 1, position will be centered such that (0,0) is in the center of the image.
 * @param deltaT			The minimum time between frames in seconds
 * @param postFrame			PostFrame function, called after the drawKernel for the frame finished, but before the framecounter is incremented
 * @param preframe			PreFrame function, called before the drawKernel for the frame is called, used to specify how much shared memory to dynamically allocate to the drawKernel.
 * @param eventFunction		The function that handles events caused during the loop.
 * @param data				User supplied data. If the drawKernel is supposed to make use of it, it must be in device-accessible memory
 * @return					The cudaError that caused the loop to stop. If cudaSuccess the loop was cancelled due to preFrame or postFrame returning a non-0 value.
 */
extern cudaError_t autoDrawCUDA(SDL_Window *win, launchDrawKernel kernel, postFrameFunc postFrame, preFrameFunc preframe, eventFunc eventFunction, void *data);

#ifdef __cplusplus
}
#endif

#endif /* SIMPLEDRAW_H_ */
