/*
 * dynmath.cuh
 *
 *  Created on: 09.11.2019
 *      Author: til
 */

#ifndef DYNMATH_CUH_
#define DYNMATH_CUH_

extern "C" {
#include <assert.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
}

typedef float scalarType;
__host__ __device__ constexpr inline scalarType operator"" _s(long double st) {
	return st;
}
__host__ __device__ constexpr inline scalarType operator"" _s(unsigned long long st) {
	return static_cast<long long>(st);
}

typedef int integerType;

__host__ __device__  constexpr inline integerType __scalar_as_integer(scalarType a) {
	assert(sizeof(integerType)==sizeof(scalarType));
	return reinterpret_cast<integerType&>(a);
}
__host__ __device__ constexpr inline scalarType __integer_as_scalar(integerType a) {
	return reinterpret_cast<scalarType&>(a);
}

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
__host__ __device__ constexpr inline scalarType asins(scalarType a) {
	return asinf(a);
}
__host__ __device__ constexpr inline scalarType acoss(scalarType a) {
	return acosf(a);
}
__host__ __device__ constexpr inline scalarType atans(scalarType a) {
	return atanf(a);
}
__host__ __device__ constexpr inline scalarType atan2s(scalarType a, scalarType b) {
	return atan2f(a,b);
}
__host__ __device__ constexpr inline scalarType logs(scalarType a) {
	return logf(a);
}
__host__ __device__ constexpr inline integerType cinv(integerType a) {
	return a>=0?a:(a^(__scalar_as_integer(.1_s)^__scalar_as_integer(-.1_s)));
}
__host__ __device__ constexpr inline integerType ftooi(scalarType a) {
	return cinv(__scalar_as_integer(a));
}
__host__ __device__ constexpr inline scalarType oitof(integerType a) {
	return __integer_as_scalar(cinv(a));
}

typedef enum : unsigned char {
	CTM=0,
	CON,
	PRM,
	ADD,
	MUL,
	POW,
	NEG,
	INV,
	SIN,
	COS,
	EXP,
	LOG
} op_t;

typedef struct __operation_t operation_t;
typedef struct __term_t term_t;

typedef struct {
	scalarType (*eval)(const operation_t &op, const scalarType **params, const term_t &term);
	scalarType (*derive)(const operation_t &op, const scalarType **params, size_t deriveVec, size_t deriveIdx, const term_t &term);
	bool iaTerm,ibTerm;
} CTM_t;

struct __operation_t {
	op_t op;
	const CTM_t *ctm;
	size_t ia,ib;
	scalarType va,vb;
	__host__ __device__ constexpr operation_t operator>>(size_t s) const {
		operation_t ret=*this;
		if(op==PRM) {}
		else {
			ret.ia+=s*(op==CTM?ctm->iaTerm:1);
			ret.ib+=s*(op==CTM?ctm->ibTerm:1);
		}
		return ret;
	}
};

struct __term_t {
	operation_t *ops;
	size_t opcount;
	__host__ __device__ scalarType eval(size_t op, const scalarType **params) const;
	__host__ __device__ scalarType derive(size_t op, const scalarType **params, size_t deriveVec, size_t deriveIdx) const;
};

#endif /* DYNMATH_CUH_ */
