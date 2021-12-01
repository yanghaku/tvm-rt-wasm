/*!
 * \file runtime/utils/cuda_common.h
 * \brief some common auxiliary definitions and functions for cuda
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_CUDA_COMMON_H
#define TVM_RT_CUDA_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_DRIVER_CALL(x)                                                                                            \
    do {                                                                                                               \
        CUresult result = x;                                                                                           \
        if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {                                            \
            const char *msg;                                                                                           \
            cuGetErrorName(result, &msg);                                                                              \
            fprintf(stderr, "CUDA Error in %s %d : %s\n", __FILE__, __LINE__, msg);                                    \
        }                                                                                                              \
    } while (0)

#define CUDA_CALL(func)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = (func);                                                                                        \
        if (e != cudaSuccess && e != cudaErrorCudartUnloading)                                                         \
            fprintf(stderr, "CUDA runtime Error in %s %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e));          \
    } while (0)

#ifdef __cplusplus
} // extern "C"
#endif
#endif // TVM_RT_CUDA_COMMON_H
