#ifndef TVM_RT_EXAMPLES_WASM_DURING_H
#define TVM_RT_EXAMPLES_WASM_DURING_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                                                   \
    long long(t0);                                                                                                     \
    GetSystemTimePreciseAsFileTime(&(t0));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)

#else // _MSC_VER

#include <sys/time.h>

#define SET_TIME(t0)                                                                                                   \
    struct timeval(t0);                                                                                                \
    gettimeofday(&(t0), NULL);

#define GET_DURING(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)

#endif // _MSC_VER

#ifdef __cplusplus
};
#endif // __cplusplus

#endif // TVM_RT_EXAMPLES_WASM_DURING_H
