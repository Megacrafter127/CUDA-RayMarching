/*
 * simpledrawCUDA.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include <cuda_runtime.h>

extern "C" {
	#include "simpledraw.h"
	#include "assert_cuda.h"
}
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cstdio>

typedef struct {
	launchDrawKernel kernel;
	preFrameFunc preframe;
	postFrameFunc postframe;
	unsigned width,height;
	void *userData;
	cudaError_t error;
	cudaStream_t stream;
	cudaEvent_t event;

	int fbc,fbi;
	// GL buffers
	GLuint*                 fb;
	GLuint*                 rb;
	// CUDA resources
	cudaGraphicsResource_t* cgr;
	cudaArray_t*            ca;
	cudaSurfaceObject_t*	co;
} cudaDrawData_t;

static inline int cudaDrawFunc(SDL_Window *win, size_t frame, cudaDrawData_t *cudata) {
	int ret=cudata->preframe(frame,cudata->userData,cudata->stream);

	if(ret) return ret;

	cuda(GraphicsMapResources(1,&cudata->cgr[cudata->fbi],cudata->stream));


	dim3 bounds = dim3(cudata->width,cudata->height,1);

	cudata->kernel(cudata->co[cudata->fbi], bounds, frame, cudata->userData, cudata->stream);

	cuda(GraphicsUnmapResources(1,&cudata->cgr[cudata->fbi],cudata->stream));

	cuda(StreamSynchronize(cudata->stream));

	glBlitNamedFramebuffer(cudata->fb[cudata->fbi],0,
			0,0,             cudata->width,cudata->height,
			0,cudata->height,cudata->width,0,
			GL_COLOR_BUFFER_BIT,
			GL_NEAREST);

	cudata->fbi++;
	cudata->fbi%=cudata->fbc;
	SDL_GL_SwapWindow(win);

	ret=cudata->postframe(frame,cudata->userData,cudata->stream);
	cudata->error=cuda(StreamSynchronize(cudata->stream));
	if(cudata->error!=cudaSuccess) ret|=1;
	return ret;
}

static cudaError_t resize(cudaDrawData_t &data, const int width, const int height) {
	cudaError_t cuda_err = cudaSuccess;
	for(int index=0; index<data.fbc; index++) {
		if(data.cgr[index]!=0) {
			cuda_err = cuda(GraphicsUnmapResources(1,&data.cgr[index],data.stream));
			if(cuda_err != cudaSuccess) return cuda_err;
		}
		if(data.co[index]!=0) {
			cuda(DestroySurfaceObject(data.co[index]));
			data.co[index]=0;
		}
	}

	cuda_err = cuda(StreamSynchronize(data.stream));

	// save new size
	data.width  = width;
	data.height = height;

	for (int index=0; index<data.fbc; index++)
	{
		// resize rbo
		glNamedRenderbufferStorage(data.rb[index],GL_RGBA32F,width,height);

		// probe fbo status
		// glCheckNamedFramebufferStatus(interop->fb[index],0);

		// register rbo
		cuda(GraphicsGLRegisterImage(&data.cgr[index],
				data.rb[index],
				GL_RENDERBUFFER,
				cudaGraphicsRegisterFlagsSurfaceLoadStore |
				cudaGraphicsRegisterFlagsWriteDiscard));
	}

	// map graphics resources
	cuda_err = cuda(GraphicsMapResources(data.fbc,data.cgr,0));
	if(cuda_err != cudaSuccess) return cuda_err;
	// get CUDA Array refernces

	cudaResourceDesc rdesc;
	rdesc.resType=cudaResourceTypeArray;

	for (int index=0; index<data.fbc; index++) {
		cuda(GraphicsSubResourceGetMappedArray(&data.ca[index],
				data.cgr[index],
				0,0));
		rdesc.res.array.array=data.ca[index];
		cuda(CreateSurfaceObject(&data.co[index],&rdesc));
	}

	// unmap graphics resources
	cuda_err = cuda(GraphicsUnmapResources(data.fbc,data.cgr,0));
	return cuda_err;
}

extern int defaultPreFrame(size_t frame, const void *data, cudaStream_t stream) {
	return 0;
}
extern int defaultPostFrame(size_t frame, void *data, cudaStream_t stream) {
	return 0;
}
extern int defaultEventFunction(SDL_Event *event, void *data) {
	return event->type == SDL_QUIT;
}

static int autoDraw(SDL_Window *win, eventFunc eventFunction, cudaDrawData_t *data) {
	if(!eventFunction) eventFunction=defaultEventFunction;
	int c=0;
	for(size_t i=0;!(c=cudaDrawFunc(win,i,data));i++) {
		if(c) return c;
		SDL_Event event;
		while(SDL_PollEvent(&event)) {
			c=eventFunction(&event,data);
			if(c) return c;
		}
	}
	return c;
}
extern cudaError_t autoDrawCUDA(SDL_Window *win, launchDrawKernel kernel, postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, void *data) {
	{
		unsigned gl_device_count;
		int gl_device_id;
		fprintf(stderr,"Obtaining devices\n");
		cudaError_t err=cudaGLGetDevices(&gl_device_count,&gl_device_id,1,cudaGLDeviceListAll);
		if(err!=cudaSuccess) return err;
		if(gl_device_count) {
			fprintf(stderr,"Set device\n");
			err=cudaSetDevice(gl_device_id);
			if(err!=cudaSuccess) return err;
		}
	}
	if(!postframe) postframe=defaultPostFrame;
	if(!preframe) preframe=defaultPreFrame;
	cudaDrawData_t cudata;
	cudata.kernel=kernel;
	cudata.preframe=preframe;
	cudata.postframe=postframe;
	cudata.userData=data;
	cudata.error=cudaStreamCreate(&cudata.stream);
	if(cudata.error!=cudaSuccess) return cudata.error;
	cudata.error=cudaEventCreateWithFlags(&cudata.event,cudaEventBlockingSync);
	if(cudata.error!=cudaSuccess) return cudata.error;

	cudata.fbc = 2;
	cudata.fbi = 0;

	// allocate arrays
	cudata.fb  = new GLuint[cudata.fbc];
	cudata.rb  = new GLuint[cudata.fbc];
	cudata.cgr = (cudaGraphicsResource_t*)calloc(cudata.fbc,sizeof(*(cudata.cgr)));
	cudata.ca  = new cudaArray_t[cudata.fbc];
	cudata.co  = new cudaSurfaceObject_t[cudata.fbc];

	// render buffer object w/a color buffer
	glCreateRenderbuffers(cudata.fbc,cudata.rb);

	// frame buffer object
	glCreateFramebuffers(cudata.fbc,cudata.fb);

	// attach rbo to fbo
	for (int index=0; index<cudata.fbc; index++) {
		glNamedFramebufferRenderbuffer(cudata.fb[index],
				GL_COLOR_ATTACHMENT0,
				GL_RENDERBUFFER,
				cudata.rb[index]);
	}

	int width,height;
	SDL_GL_GetDrawableSize(win,&width,&height);
	cudata.error=resize(cudata,width,height);
	if(cudata.error!=cudaSuccess) return cudata.error;

	autoDraw(win,eventFunction,&cudata);

	cudaStreamDestroy(cudata.stream);
	return cudata.error;
}
