//
//
//

#include <cuda_runtime.h>
#include <stdbool.h>

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

//
//
//

//
// Beware that NVCC doesn't work with C files and __VA_ARGS__
//

extern cudaError_t cuda_assert(const cudaError_t code, const char* const file, const int line, const bool abort);

#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__, false);

//
//
//

#ifdef __cplusplus
}
#endif
