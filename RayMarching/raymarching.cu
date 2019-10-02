/*
 * raymarching.cu
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#include "raymarching.cuh"

#include <cmath>
#include <cassert>

__device__ static void marchRay(argb &pixel, uint3 pos, size_t frame, const void *data) {
	__shared__ world_t world;
	if(threadIdx.x|threadIdx.y|threadIdx.z==0) {
		memcpy(&world,data,sizeof(world_t));
	}
	__syncthreads();
	register scalarType totalDist=0,divergence;
	register vectorType start=world.camera.pos,ray=world.camera.rays(divergence,pos,frame);
	register scalarType rayLen=norm(ray);
	register floatColor_t color;
	color.r=0;
	color.g=0;
	color.b=0;
	color.a=1.0f/0xFF;
	register size_t step=0;
	for(;totalDist*SCL_EPSILON < world.maxErr;step++) {
		register scalarType minDist=INFINITY;
		register size_t minShape=world.shapeCount;
		for(register size_t i=0;i<world.shapeCount;i++) {
			register scalarType dist=world.shapes[i].getDistance(start,frame);
			//if(dbg) printf("Step:    %lld\nShape:   %3lld %f\nClosest: %3zd %f\n\n",step,i,dist,minShape,minDist);
			if(dist<minDist) {
				minDist=dist;
				minShape=i;
			}
		}
		if(minShape==world.shapeCount) break;
		assert(minShape<world.shapeCount);
		//if(minDist>maxf(totalDist,100)) break;
		const register scalarType cdiv=totalDist*divergence;
		if(cdiv>minDist) {
			floatColor_t shapeColor=world.shapes[minShape].getColor(start,minDist,cdiv,frame,step);
			if(minDist>0) shapeColor.a*=1-minDist/cdiv;
			if(shapeColor.a<0) shapeColor.a=0;
			if(shapeColor.a>1) shapeColor.a=1;
			color+=shapeColor;
			if(color.a*0xFF>=0xFE) break;
		}
		if(minDist<=0) break;
		totalDist+=minDist;
		start=start+ray*(minDist/rayLen);
	}
	color.r*=color.a;
	color.g*=color.a;
	color.b*=color.a;
	color.a=1;
	pixel=(argb)color;
}
__managed__ static cudaFunc rayMarch=marchRay;

cudaError_t autoRenderShapes(SDL_Surface *surface, world_t *world, float deltaT, postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, unsigned x_threads, unsigned y_threads) {
	cudaError_t err=autoDrawCUDA(surface,rayMarch,deltaT,postframe,preframe,eventFunction,world,x_threads,y_threads);
	return err;
}
