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

#include <sys/time.h>

#define DURING_PRINT(t1, t0, msg)                                                                                      \
    fprintf(stderr, "%s: %lf ms\n", msg,                                                                               \
            (double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.f);

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
