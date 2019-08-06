/*
 * main.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "raymarching.cuh"

#include <cmath>
#include <ctime>

#define CHUNK_SIZE 16

#define IMG_CHUNKS_WIDTH 50
#define IMG_WIDTH (IMG_CHUNKS_WIDTH*CHUNK_SIZE)
#define IMG_CHUNKS_HEIGHT 40
#define IMG_HEIGHT (IMG_CHUNKS_HEIGHT*CHUNK_SIZE)

#if IMG_WIDTH>IMG_HEIGHT
#define IMG_MIN IMG_HEIGHT
#define IMG_MAX IMG_WIDTH
#else
#define IMG_MAX IMG_HEIGHT
#define IMG_MIN IMG_WIDTH
#endif

static SDL_Surface *surf;

__managed__ static world_t world;

#define SHAPE_COUNT 9

__constant__ static shape_t shapes[SHAPE_COUNT];

typedef struct {
	floatColor_t color;
} simpleColor_t;

typedef struct : simpleColor_t {
	float3 center;
	float radius;
} sphereData_t;

__constant__ static sphereData_t shapeData[SHAPE_COUNT];

__host__ __device__ inline float3 operator-(float3 a, float3 b) {
	return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}

__device__ float sphereDistance(const void *shapeData, float3 point, size_t frame) {
	register const sphereData_t * const sphere=static_cast<const sphereData_t*>(shapeData);
	register float3 diff=point-sphere->center;
	return norm3df(diff.x,diff.y,diff.z)-sphere->radius;
}
__managed__ distanceFunc sphereDistAddr=sphereDistance;

__device__ float cubeDistance(const void *shapeData, float3 point, size_t frame) {
	register const sphereData_t * const sphere=static_cast<const sphereData_t*>(shapeData);
	register float3 diff=point-sphere->center;
	diff.x=fabs(diff.x);
	diff.y=fabs(diff.y);
	diff.z=fabs(diff.z);
	if(diff.y>diff.x) diff.x=diff.y;
	if(diff.z>diff.x) diff.x=diff.z;
	return diff.x-sphere->radius;
}
__managed__ distanceFunc cubeDistAddr=cubeDistance;

__device__ static floatColor_t glowColor(const void *shapeData, float3 point, float distance, float divergence, size_t frame, size_t steps) {
	register floatColor_t color=static_cast<const simpleColor_t*>(shapeData)->color;
	const register float stepf=1-1/(steps*.0625f+1),divf=1-distance/divergence;
	color.r*=divf*stepf;
	color.g*=divf*stepf;
	color.b*=divf*stepf;
	return color;
}
__managed__ colorFunc glowColorAddr=glowColor;

int handleError(cudaError_t err) {
	if(err==cudaSuccess) return 0;
	fprintf(stderr,"%s: %s\n",cudaGetErrorName(err),cudaGetErrorString(err));
	return 1;
}

inline float3 operator/(float3 a, float b) {
	return make_float3(a.x/b,a.y/b,a.z/b);
}

inline float3 operator-(float3 a) {
	return make_float3(-a.x,-a.y,-a.z);
}

static int postFrame(size_t frame, void *data) {
	const clock_t t=clock();
	const register float dn=-1.0f/(IMG_MAX-1),df=.25f,drad=.2f/CLOCKS_PER_SEC;
	const register float s=sinf(t*drad),c=cosf(t*drad);
	world.camera.dx=make_float3(s*dn,0,-c*dn);
	world.camera.pos=make_float3(-c*df,0,-s*df);
	world.camera.face=-world.camera.pos-world.camera.dx*IMG_WIDTH/2-world.camera.dy*IMG_HEIGHT/2;
	return 0;
}

const static float xs[SHAPE_COUNT]={ 10, 10,  0,-10,-10,-10,  0, 10};

int main() {
	world.camera.dy=make_float3(0,-1.0f/(IMG_MAX-1),0);
	postFrame(0,NULL);
	cudaGetSymbolAddress((void**)&(world.shapes),shapes);
	world.shapeCount=SHAPE_COUNT;

	sphereData_t *sd;
	cudaGetSymbolAddress((void**)&sd,shapeData);

	for(int i=0;i<SHAPE_COUNT;i++,sd++) {
		cudaMemcpyToSymbol(shapes,&glowColorAddr,sizeof(colorFunc),sizeof(shape_t)*i+offsetof(shape_t,colorFunction));
		cudaMemcpyToSymbol(shapes,i%2?&sphereDistAddr:&cubeDistAddr,sizeof(distanceFunc),sizeof(shape_t)*i+offsetof(shape_t,distanceFunction));
		cudaMemcpyToSymbol(shapes,&sd,sizeof(void*),sizeof(shape_t)*i+offsetof(shape_t,shapeData));
	}
	sphereData_t sample;

	sample.center=make_float3(0,0,0);
	sample.color.a = 1;
	sample.color.r = .75f;
	sample.color.g = 0;
	sample.color.b = .75f;
	sample.radius = .0625f;
	cudaMemcpyToSymbol(shapeData,&sample,sizeof(sphereData_t));

	for(size_t i=0;i<SHAPE_COUNT-1;i++) {
		sample.center=make_float3(xs[i],0,xs[(i+2)%(SHAPE_COUNT-1)]);
		sample.color.a = 1;
		sample.color.r = i%2?0:1;
		sample.color.g = (i/2)%2?0:1;
		sample.color.b = (i/4)%2?0:1;
		if(i%8==7) {
			sample.color.r=.125f;
			sample.color.g=.125f;
			sample.color.b=.125f;
		}
		sample.radius = i%2*3+i%3+1;
		cudaMemcpyToSymbol(shapeData,&sample,sizeof(sphereData_t),sizeof(sphereData_t)*(i+1));
	}

	surf=createSurface(IMG_WIDTH,IMG_HEIGHT,"Raymarching Test");
	int ret=handleError(autoRenderShapes(surf,&world,0,postFrame));
	destroySurface(surf);
	return ret;
}