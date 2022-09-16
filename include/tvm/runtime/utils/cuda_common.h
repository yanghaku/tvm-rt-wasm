/*!
 * \file runtime/utils/cuda_common.h
 * \brief some common auxiliary definitions and functions for cuda
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_CUDA_COMMON_H
#define TVM_RT_WASM_CUDA_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/utils/common.h>

#if USE_CUDA // USE_CUDA = 1

#include <cuda.h>

#define CUDA_DRIVER_CALL(x)                                                                                            \
    do {                                                                                                               \
        CUresult result = (x);                                                                                         \
        if (unlikely(result != CUDA_SUCCESS)) {                                                                        \
            const char *msg;                                                                                           \
            cuGetErrorString(result, &msg);                                                                            \
            SET_ERROR_RETURN(-1, "CUDA Driver Call Error: %s", msg);                                                   \
        }                                                                                                              \
    } while (0)

#define CUDA_DRIVER_CALL_NULL(x)                                                                                       \
    do {                                                                                                               \
        CUresult result = (x);                                                                                         \
        if (unlikely(result != CUDA_SUCCESS)) {                                                                        \
            const char *msg;                                                                                           \
            cuGetErrorString(result, &msg);                                                                            \
            SET_ERROR_RETURN(NULL, "CUDA Driver Call Error: %s", msg);                                                 \
        }                                                                                                              \
    } while (0)

#else

#define CUDA_NOT_SUPPORTED()                                                                                           \
    do {                                                                                                               \
        fprintf(stderr, "CUDA library is not supported! you can recompile from source and set USE_CUDA option ON\n");  \
        exit(-1);                                                                                                      \
    } while (0)

#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CUDA_COMMON_H
