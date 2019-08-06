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
	if(threadIdx.x|threadIdx.y==0) {
		memcpy(&world,data,sizeof(world_t));
	}
	__syncthreads();
	register float3 start=world.camera.pos,ray=world.camera.face+pos.x*world.camera.dx+pos.y*world.camera.dy;
	register float rayLen=normf3(ray),totalDist=0,divergence=(normf3(world.camera.dx)+normf3(world.camera.dy))*.5f/rayLen;
	/*if(pos.x==0 && pos.y==0) {
		printff3("Start",start);
		printff3("Face",world->camera.face);
		printff3("Dx",world->camera.dx);
		printff3("Dy",world->camera.dy);
		printff3("Ray",ray);
		printf("\n");
	}*/
	register floatColor_t color;
	color.r=0;
	color.g=0;
	color.b=0;
	color.a=1.0f/0xFF;
	register floatColor_t black=color;
	register size_t step=0;
	do{
		float minDist=INFINITY;
		size_t minShape=-1;
		for(size_t i=0;i<world.shapeCount;i++) {
			float dist=world.shapes[i].getDistance(start,frame);
			if(dist<minDist) {
				minDist=dist;
				minShape=i;
			}
		}
		if(minShape==-1) break;
		if(minDist>maxf(totalDist,100)) break;
		const register float cdiv=totalDist*divergence;
		if(cdiv>minDist) {
			//printf("Hit\n");
			floatColor_t shapeColor=world.shapes[minShape].getColor(start,minDist,cdiv,frame,step);
			if(minDist>0) shapeColor.a*=1-minDist/cdiv;
			if(shapeColor.a<0) shapeColor.a=0;
			if(shapeColor.a>1) shapeColor.a=1;
			color+=shapeColor;
		}
		if(minDist<=0) break;
		totalDist+=minDist;
		start=start+ray*(minDist/rayLen);
		step++;
	} while(color.a*0xFF<=0xFE);
	black.a=1-color.a;
	color+=black;
	pixel=(argb)color;
}
__managed__ static cudaFunc rayMarch=marchRay;

cudaError_t autoRenderShapes(SDL_Surface *surface, world_t *world, float deltaT, postFrameFunc postframe, preFrameFunc preframe, eventFunc eventFunction, unsigned x_threads, unsigned y_threads) {
	cudaError_t err=autoDrawCUDA(surface,rayMarch,deltaT,postframe,preframe,eventFunction,world,x_threads,y_threads);
	return err;
}
