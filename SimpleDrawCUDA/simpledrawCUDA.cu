/*
 * simpledrawCUDA.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "simpledrawCUDA.cuh"

#include <cassert>
#include <cmath>
#include <ctime>

__host__ __device__ inline uint3 operator*(uint3 a, dim3 b) {
	return make_uint3(a.x*b.x,a.y*b.y,a.z*b.z);
}

__host__ __device__ inline uint3 operator*(unsigned b, uint3 a) {
	return make_uint3(a.x*b,a.y*b,a.z*b);
}

__host__ __device__ inline uint3 operator+(uint3 a, uint3 b) {
	return make_uint3(a.x+b.x,a.y+b.y,a.z+b.z);
}

__host__ __device__ inline int3 operator-(uint3 a, uint3 b) {
	return make_int3(a.x-b.x,a.y-b.y,a.z-b.z);
}

__host__ __device__ inline uint3 operator/(uint3 a, int b) {
	return make_uint3(a.x/b,a.y/b,a.z/b);
}


__global__ void drawKernel(void *pixels, size_t pixelPitch, cudaFunc func, int center, uint3 bounds, size_t frame, const void * data) {
	const uint3 pos = threadIdx+blockIdx*blockDim;
	void *row=((char*)pixels)+pixelPitch*pos.y;
	func(((argb*)row)[pos.x],pos-center*bounds/2,frame,data);
}

#define CHUNKIFY(len,chunk) (len+(chunk-len%chunk)%chunk)
#define CHUNKCOUNT(len,chunk) (CHUNKIFY(len,chunk)/chunk)

typedef struct {
	void *pixels;
	size_t pixelPitch;
	cudaFunc func;
	int center;
	preFrameFunc preframe;
	postFrameFunc postframe;
	unsigned x_threads,y_threads,z_threads,z_blocks;
	size_t dyn_alloc;
	void *userData;
	cudaError_t error;
	cudaStream_t stream;
} cudaDrawData_t;

static int cudaDrawFunc(SDL_Surface *surf, size_t frame, void *data) {
	cudaDrawData_t *cudata=(cudaDrawData_t*)data;
	int ret=cudata->preframe(frame,cudata->userData,cudata->z_blocks,cudata->z_threads,cudata->dyn_alloc);
	if(ret) return ret;
	const dim3 blocks=dim3(CHUNKCOUNT(surf->w,cudata->x_threads),CHUNKCOUNT(surf->h,cudata->y_threads),cudata->z_blocks),
		 threads=dim3(cudata->x_threads,cudata->y_threads,cudata->z_threads);
	drawKernel<<<blocks,threads,cudata->dyn_alloc,cudata->stream>>>(cudata->pixels,cudata->pixelPitch,cudata->func,cudata->center,make_uint3(surf->w,surf->h,0),frame,cudata->userData);
	cudata->error=cudaStreamSynchronize(cudata->stream);
	if(cudata->error!=cudaSuccess) return 1;
	cudata->error=cudaMemcpy2DAsync(surf->pixels,surf->pitch,cudata->pixels,cudata->pixelPitch,surf->w*surf->format->BytesPerPixel,surf->h,cudaMemcpyDeviceToHost,cudata->stream);
	if(cudata->error!=cudaSuccess) return 1;
	ret=cudata->postframe(frame,cudata->userData);
	cudata->error=cudaStreamSynchronize(cudata->stream);
	if(cudata->error!=cudaSuccess) ret|=1;
	return ret;
}

int defaultPreFrame(size_t frame, const void *data, unsigned &z_blocks, unsigned &z_threads, size_t &dyn_alloc) {
	return 0;
}
int defaultPostFrame(size_t frame, void *data) {
	return 0;
}

cudaError_t autoDrawCUDA(SDL_Surface *surf, cudaFunc func, int center, float deltaT, postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, void *data, unsigned x_threads, unsigned y_threads) {
	assert(surf->format->BitsPerPixel==32);
	assert(surf->format->BytesPerPixel==sizeof(argb));
	assert(surf->format->Amask==0xFF000000);
	assert(surf->format->Rmask==0x00FF0000);
	assert(surf->format->Gmask==0x0000FF00);
	assert(surf->format->Bmask==0x000000FF);
	if(!postframe) postframe=defaultPostFrame;
	if(!preframe) preframe=defaultPreFrame;
	if(!x_threads) x_threads=16;
	if(!y_threads) y_threads=16;
	cudaDrawData_t cudata;
	cudata.func=func;
	cudata.center=center;
	cudata.preframe=preframe;
	cudata.x_threads=x_threads;
	cudata.y_threads=y_threads;
	cudata.z_threads=1;
	cudata.z_blocks=1;
	cudata.dyn_alloc=0;
	cudata.postframe=postframe;
	cudata.userData=data;
	cudata.error=cudaMallocPitch(&cudata.pixels,&cudata.pixelPitch,CHUNKIFY(surf->w,x_threads)*surf->format->BytesPerPixel,CHUNKIFY(surf->h,y_threads));
	if(cudata.error!=cudaSuccess) return cudata.error;
	cudata.error=cudaStreamCreate(&cudata.stream);
	if(cudata.error!=cudaSuccess) return cudata.error;

	autoDraw(surf,cudaDrawFunc,deltaT,eventFunction,&cudata);

	cudaStreamDestroy(cudata.stream);
	cudaFree(cudata.pixels);
	return cudata.error;
}
