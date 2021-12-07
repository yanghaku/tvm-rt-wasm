/*!
 * \file runtime/utils/common.h
 * \brief some common auxiliary definitions and functions
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_COMMON_H
#define TVM_RT_WASM_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

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

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

extern char global_buf[];

#define GLOBAL_BUF_SIZE 1024

#undef NDEBUG
#ifdef NDEBUG // release
#define SET_ERROR_RETURN(err, fmt, ...)                                                                                \
    do {                                                                                                               \
        sprintf(global_buf, fmt, ##__VA_ARGS__);                                                                       \
        return (err);                                                                                                  \
    } while (0)
#else
#define DBG(fmt, ...)                                                                                                  \
    do {                                                                                                               \
        fprintf(stderr, "function[%s]-line[%d]: " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__);                         \
    } while (0)

#define SET_ERROR_RETURN(err, fmt, ...)                                                                                \
    do {                                                                                                               \
        sprintf(global_buf, "function[%s]-line[%d]: " fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__);                     \
        return (err);                                                                                                  \
    } while (0)
#endif

#define SET_ERROR(fmt, ...)                                                                                            \
    do {                                                                                                               \
        sprintf(global_buf, fmt, ##__VA_ARGS__);                                                                       \
    } while (0)

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#ifdef __STDC_VERSION__

#if __STDC_VERSION__ >= 199901L
#define INLINE static inline
#else
#define INLINE static
#endif

#else // c89 c90
#define INLINE static inline
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_COMMON_H
