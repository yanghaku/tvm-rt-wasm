/**
 * @file cuda_common.h
 * @brief Some common auxiliary definitions and functions for cuda.
 */

#ifndef TVM_RT_WASM_ACCELERATORS_CUDA_CUDA_COMMON_H_INCLUDE_
#define TVM_RT_WASM_ACCELERATORS_CUDA_CUDA_COMMON_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda.h>
#include <utils/common.h>

#define CUDA_DRIVER_CALL(x)                                                                        \
    do {                                                                                           \
        CUresult result = (x);                                                                     \
        if (unlikely(result != CUDA_SUCCESS)) {                                                    \
            const char *msg;                                                                       \
            cuGetErrorString(result, &msg);                                                        \
            TVM_RT_SET_ERROR_RETURN(-1, "CUDA Driver Call Error: %s", msg);                        \
        }                                                                                          \
    } while (0)

#define CUDA_DRIVER_CALL_OR_GOTO(x, label)                                                         \
    do {                                                                                           \
        CUresult result = (x);                                                                     \
        if (unlikely(result != CUDA_SUCCESS)) {                                                    \
            const char *msg;                                                                       \
            cuGetErrorString(result, &msg);                                                        \
            TVM_RT_SET_ERROR_AND_GOTO(label, "CUDA Driver Call Error: %s", msg);                   \
        }                                                                                          \
    } while (0)

#define CUDA_DRIVER_CALL_OR_GOTO_WITH_STATUS(x, label)                                             \
    do {                                                                                           \
        CUresult result = (x);                                                                     \
        if (unlikely(result != CUDA_SUCCESS)) {                                                    \
            status = -1;                                                                           \
            const char *msg;                                                                       \
            cuGetErrorString(result, &msg);                                                        \
            TVM_RT_SET_ERROR_AND_GOTO(label, "CUDA Driver Call Error: %s", msg);                   \
        }                                                                                          \
    } while (0)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_ACCELERATORS_CUDA_CUDA_COMMON_H_INCLUDE_
