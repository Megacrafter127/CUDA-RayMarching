/*
 * vecops.cuh
 *
 *  Created on: 02.10.2019
 *      Author: Megacrafter127
 */

#ifndef VECOPS_CUH_
#define VECOPS_CUH_

#include <cmath>
#include <cfloat>

typedef float scalarType;

__host__ __device__ constexpr inline scalarType nans(const char *c) {
	return nanf(c);
}
#define SCL_EPSILON FLT_EPSILON

__host__ __device__ constexpr inline scalarType sins(scalarType a) {
	return sinf(a);
}
__host__ __device__ constexpr inline scalarType coss(scalarType a) {
	return cosf(a);
}
__host__ __device__ constexpr inline scalarType tans(scalarType a) {
	return tanf(a);
}

typedef float3 vectorType;

__host__ __device__ constexpr vectorType &operator*=(vectorType &a, scalarType b) {
	a.x*=b;
	a.y*=b;
	a.z*=b;
	return a;
}

__host__ __device__ constexpr vectorType operator*(vectorType a, scalarType b) {
	return a*=b;
}

__host__ __device__ constexpr vectorType operator*(scalarType a, vectorType b) {
	return b*a;
}

__host__ __device__ constexpr scalarType operator*(vectorType a, vectorType b) {
	scalarType ret=0;
	ret+=a.x*b.x;
	ret+=a.y*b.y;
	ret+=a.z*b.z;
	return ret;
}

__host__ __device__ constexpr vectorType &operator+=(vectorType &a, vectorType b) {
	a.x+=b.x;
	a.y+=b.y;
	a.z+=b.z;
	return a;
}

__host__ __device__ constexpr vectorType operator+(vectorType a, vectorType b) {
	return a+=b;
}

__device__ __host__ inline scalarType norm(vectorType vector) {
#ifndef __CUDA_ARCH__
	return hypotf(vector.x,hypotf(vector.y,vector.z));
#else
	return norm3df(vector.x,vector.y,vector.z);
#endif
}

__device__ __host__ constexpr bool operator==(uint3 a, uint3 b) {
	return  a.x==b.x &&
			a.y==b.y &&
			a.z==b.z;
}

__host__ __device__ constexpr vectorType &operator/=(vectorType &a, scalarType b) {
	a.x/=b;
	a.y/=b;
	a.z/=b;
	return a;
}

__host__ __device__ constexpr vectorType operator/(vectorType a, scalarType b) {
	return a/=b;
}

__host__ __device__ inline vectorType normv(vectorType a) {
	return a/norm(a);
}

__host__ __device__ constexpr vectorType operator-(vectorType a) {
	a.x=-a.x;
	a.y=-a.y;
	a.z=-a.z;
	return a;
}

__host__ __device__ constexpr vectorType &operator-=(vectorType &a, vectorType b) {
	return a+=-b;
}

__host__ __device__ constexpr vectorType operator-(vectorType a, vectorType b) {
	return a-=b;
}

__host__ __device__ constexpr vectorType cross(vectorType a, vectorType b) {
	vectorType ret{};
	ret.x=a.y*b.z-a.z*b.y;
	ret.y=a.z*b.x-a.x*b.z;
	ret.z=a.x*b.y-a.y-b.x;
	return ret;
}

__host__ __device__ inline vectorType rotate(vectorType vec, vectorType axis, scalarType angle) {
	const register vectorType aligned=axis*(vec*axis/(axis*axis));
	const register vectorType ortho=vec-aligned;
	const register vectorType w=-normv(cross(axis,ortho));
	return aligned+norm(ortho)*(normv(ortho)*coss(angle)+w*sins(angle));
}

#endif /* VECOPS_CUH_ */
