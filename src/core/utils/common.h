/*!
 * \file utils/common.h
 * \brief some common auxiliary definitions and functions
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_COMMON_H
#define TVM_RT_WASM_COMMON_H

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
#define DURING_PRINT(t1, t0, msg)                                                                                      \
    do {                                                                                                               \
    } while (0)

#else

#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                                                   \
    long long(t0);                                                                                                     \
    GetSystemTimePreciseAsFileTime(&(t0));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)

#else

#include <sys/time.h>

#define SET_TIME(t0)                                                                                                   \
    struct timeval(t0);                                                                                                \
    gettimeofday(&(t0), NULL);

#define GET_DURING(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)

#endif

#define DURING_PRINT(t1, t0, msg)                                                                                      \
    do {                                                                                                               \
        fprintf(stderr, "%s: %lf ms\n", msg, GET_DURING(t1, t0));                                                      \
    } while (0)

#endif // NDEBUG

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

extern char global_buf[];

#define GLOBAL_BUF_SIZE 4096

#ifdef NDEBUG // release

#define DBG(fmt, ...)                                                                                                  \
    do {                                                                                                               \
    } while (0)

#else

#define DBG(fmt, ...)                                                                                                  \
    do {                                                                                                               \
        fprintf(stderr, "function[%s]-line[%d]: " fmt "\n", __FUNCTION__, __LINE__, ##__VA_ARGS__);                    \
    } while (0)

#endif // NDEBUG

#define TVM_RT_SET_ERROR(fmt, ...)                                                                                     \
    do {                                                                                                               \
        DBG(fmt, ##__VA_ARGS__);                                                                                       \
        sprintf(global_buf, fmt "\n", ##__VA_ARGS__);                                                                  \
    } while (0)

#define TVM_RT_SET_ERROR_AND_GOTO(label, fmt, ...)                                                                     \
    do {                                                                                                               \
        TVM_RT_SET_ERROR(fmt, ##__VA_ARGS__);                                                                          \
        goto label;                                                                                                    \
    } while (0)

#define TVM_RT_SET_ERROR_RETURN(err, fmt, ...)                                                                         \
    do {                                                                                                               \
        TVM_RT_SET_ERROR(fmt, ##__VA_ARGS__);                                                                          \
        return (err);                                                                                                  \
    } while (0)

#define TVM_RT_NOT_IMPLEMENT(err) TVM_RT_SET_ERROR_RETURN(err, "%s is not implemented yet.\n", __FUNCTION__)

#define TVM_RT_FEATURE_NOT_ON(feature, option)                                                                         \
    do {                                                                                                               \
        fprintf(stderr, "%s is not supported! You can recompile library from source with `%s`=`ON`\n", feature,        \
                option);                                                                                               \
        exit(-1);                                                                                                      \
    } while (0)

#define CHECK_INPUT_POINTER(p, ret, var)                                                                               \
    do {                                                                                                               \
        if (unlikely((p) == NULL)) {                                                                                   \
            TVM_RT_SET_ERROR_RETURN(ret, "Invalid argument: %s cannot be NULL.", var);                                 \
        }                                                                                                              \
    } while (0)

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#ifdef __STDC_VERSION__

#if __STDC_VERSION__ >= 199901L
#define INLINE static inline
#else
#define INLINE static
#endif

#else  // c89 c90
#define INLINE static inline
#endif // __STDC_VERSION__

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_COMMON_H
