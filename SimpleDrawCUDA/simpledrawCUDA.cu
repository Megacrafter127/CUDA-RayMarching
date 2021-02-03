/*
 * simpledrawCUDA.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "simpledrawCUDA.cuh"
extern "C" {
#include "assert_cuda.h"
}
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cstdio>

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

__host__ __device__ inline int3 operator-(int3 a, int3 b) {
	return make_int3(a.x-b.x,a.y-b.y,a.z-b.z);
}

__host__ __device__ inline uint3 operator/(uint3 a, int b) {
	return make_uint3(a.x/b,a.y/b,a.z/b);
}

__host__ __device__ inline int3 &operator/=(int3 &a, int b) {
	a.x/=b;
	a.y/=b;
	a.z/=b;
	return a;
}

__global__ void drawKernel(cudaSurfaceObject_t surf, cudaFunc func, int center, uint3 bounds, size_t frame, const void * data) {
	const uint3 pos = threadIdx+blockIdx*blockDim;
	color_t color = func(pos-center*bounds/2, frame, data);
	if(pos.z == 0) surf2Dwrite(toFloat4(color), surf, pos.x*sizeof(float4), pos.y, cudaBoundaryModeZero);
}

#define CHUNKIFY(len,chunk) (len+(chunk-len%chunk)%chunk)
#define CHUNKCOUNT(len,chunk) (CHUNKIFY(len,chunk)/chunk)

typedef struct {
	cudaFunc func;
	int center;
	preFrameFunc preframe;
	postFrameFunc postframe;
	dim3 threads;
	unsigned z_blocks,width,height;
	size_t dyn_alloc;
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
	int ret=cudata->preframe(frame,cudata->userData,cudata->threads,cudata->z_blocks,cudata->dyn_alloc,cudata->stream);

	if(ret) return ret;

	cuda(GraphicsMapResources(1,&cudata->cgr[cudata->fbi],cudata->stream));

	//cuda(BindSurfaceToArray(surf,cudata->ca[cudata->fbi]));
	const dim3 blocks=dim3(CHUNKCOUNT(cudata->width,cudata->threads.x),CHUNKCOUNT(cudata->height,cudata->threads.y),cudata->z_blocks);

	drawKernel<<<blocks,cudata->threads,cudata->dyn_alloc,cudata->stream>>>(cudata->co[cudata->fbi],cudata->func,cudata->center,make_uint3(cudata->width,cudata->height,0),frame,const_cast<const void*>(cudata->userData));

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

int defaultPreFrame(size_t frame, const void *data, dim3 &threads, unsigned &z_blocks, size_t &dyn_alloc, cudaStream_t stream) {
	return 0;
}
int defaultPostFrame(size_t frame, void *data, cudaStream_t stream) {
	return 0;
}
int defaultEventFunction(SDL_Event *event, void *data) {
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
cudaError_t autoDrawCUDA(SDL_Window *win, cudaFunc func, int center, postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, void *data, unsigned x_threads, unsigned y_threads) {
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
	if(!x_threads) x_threads=16;
	if(!y_threads) y_threads=16;
	cudaDrawData_t cudata;
	cudata.func=func;
	cudata.center=center;
	cudata.preframe=preframe;
	cudata.threads=dim3(x_threads,y_threads,1);
	cudata.z_blocks=1;
	cudata.dyn_alloc=0;
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

constexpr static uint16_t ovl(uint16_t a=0, uint16_t b=0, uint16_t c=0, uint16_t d=0, uint16_t e=0, uint16_t f=0, uint16_t g=0, uint16_t h=0, uint16_t i=0, uint16_t j=0, uint16_t k=0, uint16_t l=0, uint16_t m=0, uint16_t n=0, uint16_t o=0, uint16_t p=0) {
	return
			a<<0|
			b<<1|
			c<<2|
			d<<3|
			e<<4|
			f<<5|
			g<<6|
			h<<7|
			i<<8|
			j<<9|
			k<<10|
			l<<11|
			m<<12|
			n<<13|
			o<<14|
			p<<15;
}

__constant__ static uint16_t numerics[7][4]={
		{ovl(1,0,1,1,1,1,1,1,1,1),		ovl(1,0,1,1,0,1,1,1,1,1),		ovl(1,0,1,1,0,1,1,1,1,1),		ovl(1,1,1,1,1,1,1,1,1,1)},
		{ovl(1,0,0,0,1,1,1,0,1,1),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(1,1,1,1,1,0,0,1,1,1)},
		{ovl(1,0,0,0,1,1,1,0,1,1),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(1,1,1,1,1,0,0,1,1,1)},
		{ovl(1,0,1,1,1,1,1,0,1,1),		ovl(0,0,1,1,1,1,1,0,1,1),		ovl(0,0,1,1,1,1,1,0,1,1),		ovl(1,1,1,1,1,1,1,1,1,1)},
		{ovl(1,0,1,0,0,0,1,0,1,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(1,1,0,1,1,1,1,1,1,1)},
		{ovl(1,0,1,0,0,0,1,0,1,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(0,0,0,0,0,0,0,0,0,0),		ovl(1,1,0,1,1,1,1,1,1,1)},
		{ovl(1,0,1,1,0,1,1,0,1,1),		ovl(1,0,1,1,0,1,1,0,1,1),		ovl(1,0,1,1,0,1,1,0,1,1),		ovl(1,1,1,1,1,1,1,1,1,1)}
};

__host__ __device__ float4 toFloat4(const color_t &col) {
	return make_float4(col.r,col.g,col.b,col.a);
}

__host__ __device__ color_t &operator+=(color_t &acc, const color_t &sum) {
	for(int i=0;i<4;i++) acc.raw[i]+=sum.raw[i];
	return acc;
}

__device__ void atomicAdd(color_t *address, const color_t &sum) {
	for(int i=0;i<4;i++) atomicAdd(&(address->raw[i]),sum.raw[i]);
}

__host__ __device__ color_t &blend(color_t &fg, const color_t &bg) {
	const register float bga=bg.a*(1-fg.a);
	for(int i=0;i<3;i++) {
		fg.raw[i]=fg.raw[i]*fg.a+bg.raw[i]*bga;
	}
	fg.a+=bga;
	return fg;
}

__host__ __device__ constexpr static int rmod(int a, int b) {
	return ((a%b)+b)%b;
}

__host__ __device__ constexpr static int rdiv(int a, int b) {
	return (a-rmod(a,b))/b;
}

__device__ static inline color_t overlayDigit(const color_t &pixel, int3 xy, color_t overlay, unsigned char digit) {
	if(xy.x<0 || xy.y<0 || xy.x>=4 || xy.y>=7) return pixel;
	if((numerics[xy.y][xy.x]&(1<<digit)) == 0) return pixel;
	return overlay;
}

__device__ color_t overlayNumbers(const color_t &pixel, int3 xy, color_t overlay, int3 pos, unsigned scale, unsigned *precisions, float *numbers, unsigned arraylen) {
	//scale and offset coordinates
	int3 relative=xy-pos;
	relative.x=rdiv(relative.x,scale);
	relative.y=rdiv(relative.y,scale);
	relative.x+=5;
	//obtain current row to determine whicih number
	const int nmb=rdiv(relative.y,8);
	if(nmb<0 || nmb>=arraylen) return pixel;
	float number=numbers[nmb];
	const unsigned &precision=precisions[nmb];
	//determine number of digits
	int len=floorf(log10f(number));
	len*=len>0;
	//get y coordinate within current tile
	relative.y=rmod(relative.y,8);
	//draw decimal dot if needed
	if(precision!=0 && relative.x==5-7*precision && relative.y==6) return overlay;
	//determine which tile we are on
	const int idx=rdiv(relative.x,7)+precision;
	//determine if this tile doesn't need to be drawn
	if(idx<-len || (idx>=0 && idx>precision)) return pixel;
	//extract the digit
	if(idx<0) number/=powf(10,-idx);
	else number*=powf(10,idx);
	const unsigned char digit=fmodf(floorf(number),10);
	//get x coordinate within the current tile
	relative.x=rmod(relative.x,7);
	//apply overlay
	return overlayDigit(pixel,relative,overlay,digit);
}
