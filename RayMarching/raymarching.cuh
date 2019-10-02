/*
 * raymarching.cuh
 *
 *  Created on: 31.07.2019
 *      Author: Megacrafter127
 */

#ifndef RAYMARCHING_CUH_
#define RAYMARCHING_CUH_

#include "vecops.cuh"

#include <simpledrawCUDA.cuh>

__host__ __device__ constexpr scalarType maxf(scalarType a, scalarType b) {
	return a>b?a:b;
}

/**
 * Device function that calculates the shortest distance between this shape and the given point
 * @param shapeData		Data associated with the shape using this distance function
 * @param point			The point to calculate the minimum distance to
 * @param frame			The framecounter
 * @return				The minimum distance between point and this shape.
 */
typedef scalarType (*distanceFunc)(const void *shapeData, vectorType point, size_t frame);

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
 * @param rayLength		The distance this ray has already traveled.
 * @param frame			The framecounter
 * @param steps			The number of steps this ray has already marched
 * @return				The color of this shape
 */
typedef floatColor_t (*colorFunc)(const void *shapeData, vectorType point, scalarType distance, scalarType rayLength, size_t frame, size_t steps);

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
	__device__ inline scalarType getDistance(vectorType point, size_t frame) const {
		return this->distanceFunction(this->shapeData,point,frame);
	}
	/**
	 * Mnemonic to call the color function with parameters.
	 * @param point			the point
	 * @param distance		the distance
	 * @param rayLength		the length the ray has already traveled
	 * @param frame			the framecounter
	 * @param steps			the number of steps this ray has already marched
	 * @return				the color
	 */
	__device__ inline floatColor_t getColor(vectorType point, scalarType distance, scalarType rayLength,size_t frame,size_t steps) const {
		return this->colorFunction(this->shapeData,point,distance,rayLength,frame,steps);
	}
} shape_t;

/**
 * Calculates the direction of the ray representing the given pixel
 * @param pixel				the coordinates of the pixel. z is usually ignored
 * @param divergenceFactor	the average increase in distance between this beam and its neighbors when the beam travels 1 unit of length.
 * @return					the direction of the ray originating from the camera's position
 */
typedef vectorType (*rayFunc)(scalarType &divergenceFactor, uint3 pixel, size_t frame);

/**
 * camera data
 */
typedef struct {
	vectorType pos;
	rayFunc rays;
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
	/**
	 * maximum permissible error
	 */
	scalarType maxErr;
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
