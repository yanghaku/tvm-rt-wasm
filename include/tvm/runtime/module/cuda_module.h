/*!
 * \file runtime/module/cuda_module.h
 * \brief define the cuda module derived from module
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_CUDA_MODULE_H
#define TVM_RT_CUDA_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/module/module.h>

#if USE_CUDA // USE_CUDA = 1
#include <tvm/runtime/utils/cuda_common.h>
#endif // USE_CUDA

/*! \brief define the cuda module derived from module */
typedef struct CUDAModule {
    MODULE_BASE_MEMBER

    // for cuda member
#if USE_CUDA // USE_CUDA = 1
    CUfunction *functions;
#endif

} CUDAModule;

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_len -1: filename, >=0: resource binary
 * @param cudaModule the out handle
 * @return 0 if successful
 */
int CUDAModuleCreate(const char *resource, int resource_len, CUDAModule **cudaModule);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CUDA_MODULE_H
