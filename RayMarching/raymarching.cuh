/*
 * raymarching.cuh
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#ifndef RAYMARCHING_CUH_
#define RAYMARCHING_CUH_

#include <simpledrawCUDA.cuh>

__host__ __device__ inline float3 operator*(float a, float3 b) {
	return make_float3(a*b.x,a*b.y,a*b.z);
}

__host__ __device__ inline float3 operator*(float3 a, float b) {
	return b*a;
}

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
	return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

__host__ __device__ inline float maxf(float a, float b) {
	return a>b?a:b;
}

__device__ __host__ inline float normf3(float3 vector) {
	return norm3df(vector.x,vector.y,vector.z);
}

/**
 * Device function that calculates the shortest distance between this shape and the given point
 * @param shapeData		Data associated with the shape using this distance function
 * @param point			The point to calculate the minimum distance to
 * @param frame			The framecounter
 * @return				The minimum distance between point and this shape.
 */
typedef float (*distanceFunc)(const void *shapeData, float3 point, size_t frame);

/**
 * Representation of color with floating point values.
 */
typedef struct __floatColor {
	constexpr const static float uVal=0xFF;
	float r,g,b,a;
	/**
	 * Blends this color with the other color.
	 * @param other		The color to blend with this color
	 */
	__host__ __device__ inline void operator+=(const struct __floatColor &other) {
		const register float s=a+other.a;
		r=r*a+other.r*other.a;
		g=g*a+other.g*other.a;
		b=b*a+other.b*other.a;
		r/=s;
		g/=s;
		b/=s;
		a=1-(1-a)*(1-other.a);
	}

	__host__ __device__ inline void operator=(argb color) {
		r=color.r/uVal;
		g=color.g/uVal;
		b=color.b/uVal;
		a=color.a/uVal;
	}

	/**
	 * Turns this color representation into 32bit argb format
	 */
	__host__ __device__ inline operator argb() const {
		register argb ret;
		ret.r=(unsigned char)rintf(uVal*r);
		ret.g=(unsigned char)rintf(uVal*g);
		ret.b=(unsigned char)rintf(uVal*b);
		ret.a=(unsigned char)rintf(uVal*a);
		return ret;
	}
} floatColor_t;

/**
 * Device function that returns the color of this shape.
 * @param shapeData		The data associated with this shape.
 * @param point			The point the ray is currently at, in case this shape has different colors depending on where it's viewed from.
 * @param distance		The distance between this shape and the point, as calculated by the distance function of this shape.
 * @param divergence	The average distance between this ray and its neighbors at this point.
 * @param frame			The framecounter
 * @param steps			The number of steps this ray has already marched
 * @return				The color of this shape
 */
typedef floatColor_t (*colorFunc)(const void *shapeData, float3 point, float distance, float divergence, size_t frame, size_t steps);

/**
 * Data structure representing a shape
 */
typedef struct {
	/**
	 * The distance function used by this shape
	 */
	distanceFunc distanceFunction;
	/**
	 * The color function used by this shape
	 */
	colorFunc colorFunction;
	/**
	 * The data associated with this shape.
	 * Must be device-accessible memory.
	 */
	void *shapeData;
	/**
	 * Mnemonic to call the distance function with parameters.
	 * @param point		the point
	 * @param frame		the framecounter
	 * @return			the distance
	 */
	__device__ inline float getDistance(float3 point, size_t frame) const {
		return this->distanceFunction(this->shapeData,point,frame);
	}
	/**
	 * Mnemonic to call the color function with parameters.
	 * @param point			the point
	 * @param distance		the distance
	 * @param divergence	the average distance between this ray and its neighbor rays at this point
	 * @param frame			the framecounter
	 * @param steps			the number of steps this ray has already marched
	 * @return				the color
	 */
	__device__ inline floatColor_t getColor(float3 point, float distance, float divergence,size_t frame,size_t steps) const {
		return this->colorFunction(this->shapeData,point,distance,divergence,frame,steps);
	}
} shape_t;

/**
 * camera data
 */
typedef struct {
	/**
	 * Camera position and direction of the top-left most ray.
	 */
	float3 pos,face;
	/**
	 * Difference between a ray and its right/bottom neighbors
	 */
	float3 dx,dy;
} camera_t;

/**
 * environment data
 */
typedef struct {
	/**
	 * camera data
	 */
	camera_t camera;
	/**
	 * shapes
	 * Must be device-accessible memory.
	 */
	shape_t *shapes;
	/**
	 * number of shapes
	 */
	size_t shapeCount;
} world_t;

/**
 *
 * @param surface
 * @param world
 * @param deltaT
 * @param postframe
 * @param preframe
 * @param eventFunction
 * @return
 */
cudaError_t autoRenderShapes(SDL_Surface *surface, world_t *world, float deltaT=.0f, postFrameFunc postframe=NULL, preFrameFunc preframe=NULL, eventFunc eventFunction=NULL, unsigned x_threads=0, unsigned y_threads=0);

#endif /* RAYMARCHING_CUH_ */
