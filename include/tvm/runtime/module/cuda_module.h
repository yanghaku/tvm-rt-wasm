/*!
 * \file runtime/module/cuda_module.h
 * \brief define the cuda module derived from module
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_CUDA_MODULE_H
#define TVM_RT_WASM_CUDA_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/cuda_common.h>

typedef struct CUDAFunctionInfo CUDAFunctionInfo;

/*! \brief define the cuda module derived from module */
typedef struct CUDAModule {
    MODULE_BASE_MEMBER

    // for cuda member
#if USE_CUDA // USE_CUDA = 1
    /*! \brief the cuda module */
    // todo: change it to support multi-GPU
    CUmodule cu_module;
    CUDAFunctionInfo *functions;
    uint32_t num_functions;
#endif

} CUDAModule;

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param cudaModule the out handle
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_CUDAModuleCreate(const char *resource, int resource_type, CUDAModule **cudaModule);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CUDA_MODULE_H
