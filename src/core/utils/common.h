/**
 * @file utils/common.h
 * @brief Some common auxiliary definitions and macros.
 */

#ifndef TVM_RT_WASM_CORE_UTILS_COMMON_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_COMMON_H_INCLUDE_

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NULL
#define NULL ((void *)0)
#endif

#ifdef NDEBUG // RELEASE

#define SET_TIME(t0)
#define GET_DURING(t1, t0)
#define DURING_PRINT(t1, t0, msg)                                                                  \
    do {                                                                                           \
    } while (0)

#else

#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                               \
    long long(t0);                                                                                 \
    GetSystemTimePreciseAsFileTime(&(t0));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)

#else

#include <sys/time.h>

#define SET_TIME(t0)                                                                               \
    struct timeval(t0);                                                                            \
    gettimeofday(&(t0), NULL);

#define GET_DURING(t1, t0)                                                                         \
    ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)

#endif

#define DURING_PRINT(t1, t0, msg)                                                                  \
    do {                                                                                           \
        fprintf(stderr, "%s: %lf ms\n", msg, GET_DURING(t1, t0));                                  \
    } while (0)

#endif // NDEBUG

// Define likely and unlikely
#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

// Define unreachable()
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#include <stddef.h>
#else // not C23
#if defined(__GNUC__) || defined(__clang__)
#define unreachable() (__builtin_unreachable())
#elif defined(_MSC_VER) // MSVC
#define unreachable() (__assume(false))
#else
#define unreachable() (abort())
#endif
#endif // C23

extern char global_buf[];

#define GLOBAL_BUF_SIZE 4096

#ifdef NDEBUG // release

#define DBG(fmt, ...)                                                                              \
    do {                                                                                           \
    } while (0)

#else

#define DBG(fmt, ...)                                                                              \
    do {                                                                                           \
        fprintf(stderr, "function[%s]-line[%d]: " fmt "\n", __FUNCTION__, __LINE__,                \
                ##__VA_ARGS__);                                                                    \
    } while (0)

#endif // NDEBUG

#define TVM_RT_SET_ERROR(fmt, ...)                                                                 \
    do {                                                                                           \
        DBG(fmt, ##__VA_ARGS__);                                                                   \
        sprintf(global_buf, fmt "\n", ##__VA_ARGS__);                                              \
    } while (0)

#define TVM_RT_SET_ERROR_AND_GOTO(label, fmt, ...)                                                 \
    do {                                                                                           \
        TVM_RT_SET_ERROR(fmt, ##__VA_ARGS__);                                                      \
        goto label;                                                                                \
    } while (0)

#define TVM_RT_SET_ERROR_RETURN(err, fmt, ...)                                                     \
    do {                                                                                           \
        TVM_RT_SET_ERROR(fmt, ##__VA_ARGS__);                                                      \
        return (err);                                                                              \
    } while (0)

#define TVM_RT_NOT_IMPLEMENT(err)                                                                  \
    TVM_RT_SET_ERROR_RETURN(err, "%s is not implemented yet.\n", __FUNCTION__)

#define TVM_RT_ACCELERATOR_NOT_ON(feature, lib)                                                    \
    do {                                                                                           \
        fprintf(stderr, "%s accelerator is not supported! You can link with the library `%s`.\n",  \
                feature, lib);                                                                     \
        exit(-1);                                                                                  \
    } while (0)

#define TVM_RT_CUDA_NOT_LINK() TVM_RT_ACCELERATOR_NOT_ON("CUDA", "tvm-rt-cuda")
#define TVM_RT_WebGPU_NOT_LINK() TVM_RT_ACCELERATOR_NOT_ON("WebGPU", "tvm-rt-webgpu")

#define CHECK_INPUT_POINTER(p, ret, var)                                                           \
    do {                                                                                           \
        if (unlikely((p) == NULL)) {                                                               \
            TVM_RT_SET_ERROR_RETURN(ret, "Invalid argument: %s cannot be NULL.", var);             \
        }                                                                                          \
    } while (0)

#define CHECK_INDEX_RANGE(max_r, index)                                                            \
    do {                                                                                           \
        if (unlikely((index) >= (max_r))) {                                                        \
            TVM_RT_SET_ERROR_RETURN(-2,                                                            \
                                    "Invalid argument: expect index in range [0,%d), but got %d",  \
                                    (max_r), (index));                                             \
        }                                                                                          \
    } while (0)

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define isdigit0to9(ch) ((ch) >= '0' && (ch) <= '9')
#define isdigit1to9(ch) ((ch) >= '1' && (ch) <= '9')

#ifdef __STDC_VERSION__

#if __STDC_VERSION__ >= 199901L
#define INLINE static inline
#else
#define INLINE static
#endif

#else  // c89 c90
#define INLINE static inline
#endif // __STDC_VERSION__

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_COMMON_H_INCLUDE_
