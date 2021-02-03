/*
 * raymarching.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "raymarching.cuh"

#include <cassert>
#include <unordered_map>

__constant__ static color_t green={0,1,0,1};

__constant__ static int3 overlayoffset={0,0,0};

__constant__ static float framedata[2];

__constant__ static unsigned precisions[2]={0,2};

__device__ static constexpr scalarType collisionDistance(scalarType totalDist,scalarType multiplier) {
	return multiplier*totalDist*SCL_EPSILON;
}

typedef struct {
	vectorType pos;
	scalarType dist,len;
} minStore_t;

__host__ __device__ inline size_t idx(uint3 i, dim3 b, size_t j) {
	return i.x+b.x*(i.y+b.y*(i.z+b.z*j));
}

__shared__ extern minStore_t mins[];

__device__ static color_t marchRay(int3 pos, size_t frame, const void *data) {
	__shared__ world_t world;
	if((threadIdx.x|threadIdx.y|threadIdx.z)==0) {
		memcpy(&world,data,sizeof(world_t));
	}
	__syncthreads();
	register scalarType totalDist=0._s,divergence;
	register vectorType start=world.camera.pos;
	register vectorType ray=world.camera.rays(divergence,pos,frame);
	ray/=norm(ray);
	register size_t step=0;

	for(size_t i=0;i<world.shapeCount;i++) {
		mins[idx(threadIdx,blockDim,i)].pos=start;
		mins[idx(threadIdx,blockDim,i)].dist=0._s;
		mins[idx(threadIdx,blockDim,i)].len=INFINITY;
	}

	register int end=0;
	for(;totalDist*SCL_EPSILON < world.maxErr;step++) {
		const register scalarType colDist=collisionDistance(totalDist,world.collisionMultiplier);
		register scalarType minDist=INFINITY;
		for(size_t i=0;i<world.shapeCount;i++) {
			const register scalarType dist=world.shapes[i].getDistance(start,frame);
			if(dist<minDist) minDist=dist;
			if(dist<colDist) end|=1;
			if(dist<mins[idx(threadIdx,blockDim,i)].len) {
				mins[idx(threadIdx,blockDim,i)].len=dist;
				mins[idx(threadIdx,blockDim,i)].pos=start;
				mins[idx(threadIdx,blockDim,i)].dist=totalDist;
			}
		}
		totalDist+=minDist;
		start+=ray*minDist;
		if(end) {
			break;
		}
	}
	register color_t color=world.background;
	for(size_t i=0;i<world.shapeCount;i++) {
		const register scalarType distance=world.shapes[i].getDistance(start,frame);
		register color_t c=world.shapes[i].getColor(frame,
				distance<collisionDistance(totalDist,world.collisionMultiplier),
				start,totalDist,distance,step,
				mins[idx(threadIdx,blockDim,i)].len,
				world.shapes[i].getDistance(mins[idx(threadIdx,blockDim,i)].pos,frame),
				mins[idx(threadIdx,blockDim,i)].dist);
		for(int i=0;i<3;i++) c.raw[i]*=c.a;
		color+=c;
	}
	if(color.a>0 || color.a<0) {
		for(int i=0;i<3;i++) color.raw[i]/=color.a;
		color.a=1;
	} else {
		for(int i=0;i<3;i++) color.raw[i]=0;
	}
	return overlayNumbers(color,pos,green,overlayoffset,2,precisions,framedata,2);
}
__managed__ static cudaFunc rayMarch=marchRay;

static std::unordered_map<const void*,preFrameFunc> pf;

static clock_t start,current=0,last;
constexpr static float cps=CLOCKS_PER_SEC;
static float frames=0,fb[2];

static int preframeF(size_t frame, const void *data, dim3 &threads, unsigned &z_blocks, size_t &dyn_shared, cudaStream_t stream) {
	clock_t c=clock();
	fb[1]=cps/(c-last);
	last=c;
	c-=start;
	c/=CLOCKS_PER_SEC;
	if(c>current) {
		fb[0]=frames;
		frames=0;
		current=c;
	}
	cudaMemcpyToSymbolAsync(framedata,&fb,sizeof(float)*2,0,cudaMemcpyHostToDevice,stream);
	frames+=1;
	dyn_shared=sizeof(minStore_t)*threads.x*threads.y*threads.z*static_cast<const world_t*>(data)->shapeCount;
	preFrameFunc f=pf[data];
	if(f) return f(frame,data,threads,z_blocks,dyn_shared,stream);
	return 0;
}

cudaError_t autoRenderShapes(SDL_Window *win, world_t *world,  postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, unsigned x_threads, unsigned y_threads) {
	pf[world]=preframe;
	cudaError_t err=autoDrawCUDA(win,rayMarch,1,postframe,preframeF,eventFunction,world,x_threads,y_threads);
	return err;
}
