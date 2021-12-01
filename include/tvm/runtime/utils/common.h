/*!
 * \file runtime/utils/common.h
 * \brief some common auxiliary definitions and functions
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_COMMON_H
#define TVM_RT_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

extern char global_buf[];

#define GLOBAL_BUF_SIZE 1024

#define SET_ERROR_RETURN(err, fmt, ...)                                                                                \
    do {                                                                                                               \
        sprintf(global_buf, "function[%s] " fmt, __FUNCTION__, ##__VA_ARGS__);                                         \
        return (err);                                                                                                  \
    } while (0)

#define SET_ERROR(fmt, ...)                                                                                            \
    do {                                                                                                               \
        sprintf(global_buf, "function[%s] " fmt, __FUNCTION__, ##__VA_ARGS__);                                         \
    } while (0)

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_COMMON_H
