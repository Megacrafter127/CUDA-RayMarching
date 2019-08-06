/*
 * simpledrawCUDA.h
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#ifndef SIMPLEDRAWCUDA_H_
#define SIMPLEDRAWCUDA_H_

extern "C" {
#include "simpledraw.h"
}

typedef struct {
	unsigned char b __attribute__((packed)),
			g __attribute__((packed)),
			r __attribute__((packed)),
			a __attribute__((packed));
} argb;

/**
 * Device function to be called on each pixel.
 * @param pixel		reference to the pixel
 * @param pos		position of the pixel on the canvas
 * @param frame		frame counter
 * @param data		user supplied data, unmodifiable to allow the use of constant device memory if needed
 */
typedef void (*cudaFunc)(argb &pixel, uint3 pos, size_t frame, const void *data);

/**
 * Host function to be called before the draw kernel launches.
 * To facilitate returning as soon as possible, inter-frame data changes should be performed post kernel finish.
 * @param frame			frame counter
 * @param data			user supplied data, unmodifiable to discourage unnecessary runtime of this function
 * @param dyn_shared	how much dynamically allocated shared memory the draw kernel should have. Initialized to 0.
 * @return				0 on success, non-0 otherwise
 */
typedef int (*preFrameFunc)(size_t frame, const void *data, unsigned &z_blocks, unsigned &z_threads, size_t &dyn_shared),
/**
 * Host function to be called after the draw kernel finished.
 * Used for inter-frame changes to user data.
 *
 * The potentially longer runtime can be afforded by running this function while the device is copying the frame back to host memory.
 * @param frame		frame counter
 * @param data		user supplied data
 * @return			0 on success, non-0 otherwise or if this is the last frame to be rendered.
 */
			(*postFrameFunc)(size_t frame, void *data);

/**
 * Default preFrame function.
 * Does nothing but return 0.
 * @param frame			frame counter
 * @param data			user supplied data, unmodifiable to discourage unnecessary runtime of this function
 * @param dyn_shared	how much dynamically allocated shared memory the draw kernel should have. Initialized to 0.
 * @return				0 on success, non-0 otherwise
 */
int defaultPreFrame(size_t frame, const void *data, unsigned &z_blocks, unsigned &z_threads, size_t &dyn_shared);
/**
 * Default postFrame function.
 * Does nothing but return 0.
 * @param frame		frame counter
 * @param data		user supplied data
 * @return			0 on success, non-0 otherwise or if this is the last frame to be rendered.
 */
int defaultPostFrame(size_t frame, void *data);

/**
 * Draw loop that automatically runs the drawKernel each frame.
 * The drawKernel will run the supplied cudaFunc once for each pixel on the surface.
 * @param surface			The surface to draw on
 * @param func				The CUDA function to draw each pixel
 * @param deltaT			The minimum time between frames in seconds
 * @param postFrame			PostFrame function, called after the drawKernel for the frame finished, but before the framecounter is incremented
 * @param preframe			PreFrame function, called before the drawKernel for the frame is called, used to specify how much shared memory to dynamically allocate to the drawKernel.
 * @param eventFunction		The function that handles events caused during the loop.
 * @param data				User supplied data. If the drawKernel is supposed to make use of it, it must be in device-accessible memory
 * @return					The cudaError that caused the loop to stop. If cudaSuccess the loop was cancelled due to preFrame or postFrame returning a non-0 value.
 */
cudaError_t autoDrawCUDA(SDL_Surface *surface, cudaFunc func, float deltaT=.0f, postFrameFunc postFrame=NULL, preFrameFunc preframe=NULL, eventFunc eventFunction=NULL, void *data=NULL, unsigned x_threads=0, unsigned y_threads=0);

#endif /* SIMPLEDRAWCUDA_H_ */
