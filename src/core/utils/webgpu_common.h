/*!
 * \file utils/webgpu_common.h
 * \brief webgpu common defination
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_WEBGPU_COMMON_H
#define TVM_RT_WASM_WEBGPU_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/common.h>

#if USE_WEBGPU // USE_WEBGPU = 1

#include <webgpu/webgpu_c_api.h>

// the error string can be got using `TVMGetLastError`
#define WGPU_CALL(x)                                                                                                   \
    do {                                                                                                               \
        int result = (x);                                                                                              \
        if (unlikely(result)) {                                                                                        \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

#else // USE_WEBGPU = 0

#define WebGPU_NOT_SUPPORTED() TVM_RT_FEATURE_NOT_ON("WebGPU", "USE_WEBGPU")

#endif // USE_WEBGPU

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_WEBGPU_COMMON_H
