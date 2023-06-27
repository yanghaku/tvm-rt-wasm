/*!
 * \file webgpu/webgpu_common.h
 * \brief webgpu common defination
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_ACCELERATORS_WEBGPU_WEBGPU_COMMON_H_INCLUDE_
#define TVM_RT_WASM_ACCELERATORS_WEBGPU_WEBGPU_COMMON_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <c_api/webgpu_c_api.h>
#include <utils/common.h>

// the error string can be got using `TVMGetLastError`
#define WGPU_CALL(x)                                                                                                   \
    do {                                                                                                               \
        int result = (x);                                                                                              \
        if (unlikely(result)) {                                                                                        \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_ACCELERATORS_WEBGPU_WEBGPU_COMMON_H_INCLUDE_
