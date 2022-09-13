#ifndef TVM_RT_EXAMPLE_WASM_TVM_ERROR_PROCESS_H
#define TVM_RT_EXAMPLE_WASM_TVM_ERROR_PROCESS_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <stdio.h>
#include <tvm/runtime/c_runtime_api.h>

#define RUN(func)                                                                                                      \
    do {                                                                                                               \
        status = (func);                                                                                               \
        if (status) {                                                                                                  \
            fprintf(stderr, "(line:%d) TVM API ERROR: %s\n", __LINE__, TVMGetLastError());                             \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#ifdef __cplusplus
};
#endif // __cplusplus

#endif // TVM_RT_EXAMPLE_WASM_TVM_ERROR_PROCESS_H
