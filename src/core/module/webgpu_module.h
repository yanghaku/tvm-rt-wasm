/*!
 * \file module/webgpu_module.h
 * \brief define the webgpu module derived from module
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_WebGPU_MODULE_H
#define TVM_RT_WASM_WebGPU_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <module/module.h>

typedef struct WebGPUFunctionInfo WebGPUFunctionInfo;

/*! \brief define the webgpu module derived from module */
typedef struct WebGPUModule {
    MODULE_BASE_MEMBER

#if USE_WEBGPU // USE_WEBGPU = 1
    /*! \brief the webgpu module */

    // todo: multi-GPU support
    WebGPUFunctionInfo *functions;
    uint32_t num_functions;
#endif

} WebGPUModule;

/*!
 * \brief create a webgpu module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param webgpuModule the out handle
 * @return >=0 if successful  (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_WebGPUModuleCreate(const char *resource, int resource_type, WebGPUModule **webgpuModule);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_WebGPU_MODULE_H
