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

/*! \brief define the cuda module derived from module */
typedef struct CUDAModule {
    MODULE_BASE_MEMBER

    // for cuda member
#if USE_CUDA // USE_CUDA = 1
    /*! \brief the cuda module */
    // todo: change it to support multi-GPU
    CUmodule cu_module;
    /*! \brief the cuda functions in cuda module */
    CUfunction *functions;
    /*! \brief the number of arguments of every function kernel */
    uint32_t *num_kernel_args;
    /*! \brief the argument storage for every function */
    void ***kernel_arg_storages;
    /*!
     * \brief the rest arguments map to thread params information
     *
     * -1: NULL; [0,3): grid_dim[] (blockIdx. ; [3,6): block_dim[] (ThreadIdx.
     *
     */
    uint8_t **func_arg_index_map;
    /*! \brief whether use dynamic shared memory */
    uint8_t *use_dyn_mem;
    /*!
     * \brief the number of the rest arguments map for every function
     *
     * \note for every wrapped function:
     *  num_func_args[func_id] + num_func_arg_map[func_id] + (use_dyn_mem==1) = num_args
     *
     *  \sa cudaWrappedFunction in cuda_module.c
     */
    uint32_t *num_func_arg_map;
    /*! \brief the number of functions */
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
int CUDAModuleCreate(const char *resource, int resource_type, CUDAModule **cudaModule);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CUDA_MODULE_H
