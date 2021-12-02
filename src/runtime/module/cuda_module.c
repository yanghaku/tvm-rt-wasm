/*!
 * \file src/runtime/module/cuda_module.c
 * \brief implement functions for cuda_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/module/cuda_module.h>

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_len -1: filename, >=0: resource binary
 * @param cudaModule the out handle
 * @return 0 if successful
 */
int CUDAModuleCreate(const char *resource, int resource_len, CUDAModule **cudaModule) {
    TrieCreate(&(*cudaModule)->env_funcs_map);
    TrieInsert((*cudaModule)->env_funcs_map, (const uint8_t *)"11", TVM_FUNCTION_HANDLE_ENCODE(0, 1));
    return 0;
}
