/*
 * vecops.cuh
 *
 *  Created on: 02.10.2019
 *      Author: Megacrafter127
 */

#ifndef VECOPS_CUH_
#define VECOPS_CUH_

#include <cfloat>

typedef float3 vectorType;
typedef float scalarType;

#define SCL_EPSILON FLT_EPSILON

__host__ __device__ constexpr vectorType operator*(vectorType a, scalarType b) {
	a.x*=b;
	a.y*=b;
	a.z*=b;
	return a;
}
__host__ __device__ constexpr vectorType operator*(scalarType a, vectorType b) {
	return b*a;
}

__host__ __device__ constexpr vectorType operator+(vectorType a, vectorType b) {
	a.x+=b.x;
	a.y+=b.y;
	a.z+=b.z;
	return a;
}

__device__ __host__ inline scalarType norm(vectorType vector) {
	return norm3df(vector.x,vector.y,vector.z);
}

__device__ __host__ constexpr bool operator==(uint3 a, uint3 b) {
	return  a.x==b.x &&
			a.y==b.y &&
			a.z==b.z;
}

#endif /* VECOPS_CUH_ */
