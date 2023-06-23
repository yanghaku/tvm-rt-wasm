/*!
 * \file utils/cuda_common.h
 * \brief some common auxiliary definitions and functions for cuda
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_CUDA_COMMON_H
#define TVM_RT_WASM_CUDA_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/common.h>

#if USE_CUDA // USE_CUDA = 1

#include <cuda.h>

#define CUDA_DRIVER_CALL(x)                                                                                            \
    do {                                                                                                               \
        CUresult result = (x);                                                                                         \
        if (unlikely(result != CUDA_SUCCESS)) {                                                                        \
            const char *msg;                                                                                           \
            cuGetErrorString(result, &msg);                                                                            \
            TVM_RT_SET_ERROR_RETURN(-1, "CUDA Driver Call Error: %s", msg);                                            \
        }                                                                                                              \
    } while (0)

#define CUDA_DRIVER_CALL_OR_GOTO(x, label)                                                                             \
    do {                                                                                                               \
        CUresult result = (x);                                                                                         \
        if (unlikely(result != CUDA_SUCCESS)) {                                                                        \
            const char *msg;                                                                                           \
            cuGetErrorString(result, &msg);                                                                            \
            TVM_RT_SET_ERROR_AND_GOTO(label, "CUDA Driver Call Error: %s", msg);                                       \
        }                                                                                                              \
    } while (0)

#else

#define CUDA_NOT_SUPPORTED() TVM_RT_FEATURE_NOT_ON("CUDA device", "USE_CUDA")

#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CUDA_COMMON_H
