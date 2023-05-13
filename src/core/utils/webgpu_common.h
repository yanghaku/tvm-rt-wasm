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

#if USE_WEBGPU // USE_WEBGPU = 1

#include <webgpu/webgpu_c_api.h>

#else // USE_WEBGPU = 0

#include <stdio.h>
#define WebGPU_NOT_SUPPORTED()                                                                                         \
    do {                                                                                                               \
        fprintf(stderr, "WebGPU is not supported! you can recompile from source and set USE_WEBGPU option ON\n");      \
        exit(-1);                                                                                                      \
    } while (0)

#endif // USE_WEBGPU

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_WEBGPU_COMMON_H
