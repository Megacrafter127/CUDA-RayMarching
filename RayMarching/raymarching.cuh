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

template<typename T> __host__ __device__ constexpr T &max(T &a, T &b) {
	return a>b?a:b;
}
template<typename T> __host__ __device__ constexpr const T &max(const T &a, const T &b) {
	return a>b?a:b;
}

/**
 * Representation of color with floating point values.
 */


/**
 * Device function that returns the color of this shape.
 * @param shapeData		The data associated with this shape.
 * @param frame			The framecounter
 * @param point			The point the ray is currently at, in case this shape has different colors depending on where it's viewed from.
 * @param rayLength		The distance this ray has already traveled.
 * @param steps			The number of steps this ray has marched.
 * @param minStep		The smallest step the ray has taken.
 * @param minStepDist	The distance the ray had already traveled when the smallest step was taken.
 * @return				The color of this shape
 */
typedef color_t (*colorFunc)(const void *shapeData, size_t frame, int collides, vectorType point, scalarType rayLength, scalarType distance, size_t steps, scalarType minStep, scalarType minStepEstimate, scalarType minStepDist);

/**
 * Device function that calculates the shortest distance between this shape and the given point
 * @param shapeData		Data associated with the shape using this distance function
 * @param point			The point to calculate the minimum distance to
 * @param frame			The framecounter
 * @return				The minimum distance between point and this shape.
 */
typedef scalarType (*distanceFunction)(const void *shapeData, vectorType point, size_t frame);

/**
 * Device function that calculates the gradient of a distanceFunction.
 * Mainly used in reflection.
 * @param shapeData		Data associated with the shape using the distanceFunction this function computes the gradient of
 * @param point			The point to calculate the gradient at
 * @param frame			The framecounter
 * @return				The gradient of the distanceFunction at the given point
 */
typedef vectorType (*gradientFunction)(const void *shapeData, vectorType point, size_t frame);

/**
 * Data structure representing a shape
 */
typedef struct {
	/**
	 * The distance function used by this shape.
	 */
	distanceFunction distanceFunc;
	/**
	 * The gradient function of distanceFunc
	 */
	gradientFunction gradientFunc;
	/**
	 * The color function used by this shape.
	 */
	colorFunc colorFunction;
	/**
	 * The data associated with this shape.
	 * Must be in device-accessible memory.
	 */
	void *shapeData;

	mutable int collided;
	/**
	 * Mnemonic to call the distance function with parameters.
	 * @param point		the point
	 * @param frame		the framecounter
	 * @return			the distance
	 */
	__device__ inline scalarType getDistance(vectorType point, size_t frame) const {
		return this->distanceFunc(this->shapeData,point,frame);
	}
	__device__ inline vectorType getGradient(vectorType point, size_t frame) const {
		return this->gradientFunc(this->shapeData,point,frame);
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
	__device__ inline color_t getColor(size_t frame, int collides, vectorType point, scalarType rayLength, scalarType dfs, size_t steps, scalarType minStep, scalarType minStepEstimate, scalarType minStepDist) const {
		return this->colorFunction(this->shapeData,frame,collides,point,rayLength,dfs,steps,minStep,minStepEstimate,minStepDist);
	}
} shape_t;

/**
 * Calculates the direction of the ray representing the given pixel
 * @param pixel				the coordinates of the pixel, with (0,0) being the center of the image. z is usually ignored.
 * @param divergenceFactor	the average increase in distance between this beam and its neighbors when the beam travels 1 unit of length.
 * @param frame				the frame to render.
 * @return					the direction of the ray originating from the camera's position
 */
typedef vectorType (*rayFunc)(scalarType &divergenceFactor, int3 pixel, size_t frame);

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
	 * background color
	 */
	color_t background;
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
	/**
	 *
	 */
	scalarType collisionMultiplier;
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
cudaError_t autoRenderShapes(SDL_Window *window, world_t *world, postFrameFunc postframe=NULL, preFrameFunc preframe=NULL, eventFunc eventFunction=NULL, unsigned x_threads=0, unsigned y_threads=0);

#endif /* RAYMARCHING_CUH_ */
