/*!
 * \file graph_executor/cuda_extension.h
 * \brief graph_executor cuda extension struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_GRAPH_CUDA_EXTENSION_H
#define TVM_RT_WASM_GRAPH_CUDA_EXTENSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <graph_executor.h>

/*!
 * \brief Allocate and initialize CUDA extension data for TVM_RT_WASM_GraphExecutor.
 *
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return 0 if successful.
 */
int TVM_RT_WASM_CUDAGraphExecutorExtensionDataCreate(TVM_RT_WASM_GraphExecutor g);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_GRAPH_CUDA_EXTENSION_H
