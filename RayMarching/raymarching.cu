/*
 * raymarching.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "raymarching.cuh"

#include <cassert>
#include <unordered_map>

//__constant__ static float4 green={0,1,0,1};

//__constant__ static int3 overlayoffset={0,0,0};

__constant__ static float framedata[2];

//__constant__ static unsigned precisions[2]={0,2};

__device__ static constexpr scalarType collisionDistance(scalarType totalDist,scalarType multiplier) {
	return multiplier*totalDist*SCL_EPSILON;
}

__host__ __device__ static constexpr float4 &operator+=(float4 &a, float4 b) {
	a.x+=b.x;
	a.y+=b.y;
	a.z+=b.z;
	a.w+=b.w;
	return a;
}

typedef struct {
	vectorType pos;
	scalarType dist,len;
} minStore_t;

__host__ __device__ inline size_t idx(uint3 i, dim3 b, size_t j) {
	return i.x+b.x*(i.y+b.y*(i.z+b.z*j));
}

__shared__ extern minStore_t mins[];

template<typename T> __device__ inline void simpleSurf2Dwrite(register T data, register cudaSurfaceObject_t surface, register int x, register int y, register cudaSurfaceBoundaryMode mode = cudaBoundaryModeZero) {
	return surf2Dwrite(data, surface, x*sizeof(T), y, mode);
}

__device__ static constexpr uint3 operator+(uint3 a, uint3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__device__ static constexpr uint3 operator*(uint3 a, dim3 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

__device__ static constexpr uint3 operator*(dim3 a, uint3 b) {
	return b*a;
}

__device__ static constexpr int3 operator-(uint3 a, dim3 b) {
	int3 ret{};
	ret.x=a.x-b.x;
	ret.y=a.y-b.y;
	ret.z=a.z-b.z;
	return ret;
}
__device__ static constexpr dim3 operator/(dim3 a, unsigned scalar) {
	a.x/=scalar;
	a.y/=scalar;
	a.z/=scalar;
	return a;
}

__global__ static void marchRay(cudaSurfaceObject_t surf, dim3 bounds, size_t frame, const void *data) {
	const register uint3 pos = threadIdx+blockDim*blockIdx;
	__shared__ world_t world;
	if((threadIdx.x|threadIdx.y|threadIdx.z)==0) {
		memcpy(&world,data,sizeof(world_t));
	}
	__syncthreads();
	register scalarType totalDist=0._s,divergence;
	register vectorType start=world.camera.pos;
	register vectorType ray=world.camera.rays(divergence,pos-bounds/2,frame);
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
	register float4 color=world.background;
	for(size_t i=0;i<world.shapeCount;i++) {
		const register scalarType distance=world.shapes[i].getDistance(start,frame);
		register float4 c=world.shapes[i].getColor(frame,
				distance<collisionDistance(totalDist,world.collisionMultiplier),
				start,totalDist,distance,step,
				mins[idx(threadIdx,blockDim,i)].len,
				world.shapes[i].getDistance(mins[idx(threadIdx,blockDim,i)].pos,frame),
				mins[idx(threadIdx,blockDim,i)].dist);
		c.x*=c.w;
		c.y*=c.w;
		c.z*=c.w;
		color+=c;
	}
	if(color.w>0 || color.w<0) {
		color.x/=color.w;
		color.y/=color.w;
		color.z/=color.w;
		color.w=1;
	} else {
		color.x=0;
		color.y=0;
		color.z=0;
	}

	simpleSurf2Dwrite(color,surf,pos.x,pos.y);
	//return overlayNumbers(color,pos,green,overlayoffset,2,precisions,framedata,2);
}

#define CHUNKIFY(len,chunk) (len+(chunk-len%chunk)%chunk)
#define CHUNKCOUNT(len,chunk) (CHUNKIFY(len,chunk)/chunk)

constexpr dim3 operator+(dim3 a, dim3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

constexpr dim3 operator-(dim3 a, dim3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

constexpr dim3 operator%(dim3 a, dim3 b) {
	a.x %= b.x;
	a.y %= b.y;
	a.z %= b.z;
	return a;
}

constexpr dim3 operator/(dim3 a, dim3 b) {
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

static std::unordered_map<const void*,dim3> threads;

static std::unordered_map<const void*,size_t> dyn_shared;

static void launchRayMarch(cudaSurfaceObject_t surface, dim3 bounds, size_t frame, const void *userData, cudaStream_t stream) {
	marchRay<<<CHUNKCOUNT(bounds,threads[userData]),threads[userData],dyn_shared[userData],stream>>>(surface,bounds,frame,userData);
}


static std::unordered_map<const void*,preFrameFunc> pf;

static clock_t start,current=0,last;
constexpr static float cps=CLOCKS_PER_SEC;
static float frames=0,fb[2];

static int preframeF(size_t frame, const void *data, cudaStream_t stream) {
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
	dyn_shared[data]=sizeof(minStore_t)*threads[data].x*threads[data].y*threads[data].z*static_cast<const world_t*>(data)->shapeCount;
	preFrameFunc f=pf[data];
	if(f) return f(frame,data,stream);
	return 0;
}

cudaError_t autoRenderShapes(SDL_Window *win, world_t *world,  postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, unsigned x_threads, unsigned y_threads) {
	pf[world]=preframe;
	threads[world]=dim3(x_threads,y_threads,1);
	cudaError_t err=autoDrawCUDA(win,launchRayMarch,postframe,preframeF,eventFunction,world);
	return err;
}
