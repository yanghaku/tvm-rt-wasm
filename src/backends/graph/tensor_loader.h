/*!
 * @file graph/tensor_loader.h
 * @brief load the DLTensor list from param blob.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_BACKENDS_GRAPH_TENSOR_LOADER_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_GRAPH_TENSOR_LOADER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <utils/tensor_helper.h>

/*!
 * @brief Parse string to DLDataType.
 * @param str the source string
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
int TVM_RT_WASM_DLDataType_ParseFromString(const char *str, DLDataType *out_type);

/*!
 * @brief Load parameters from stream reader.
 * @param graph The instance of TVM_RT_WASM_GraphExecutor.
 * @param reader The stream reader instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorLoadParamsFromReader(TVM_RT_WASM_GraphExecutor graph,
                                                  StreamReader *reader);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_GRAPH_TENSOR_LOADER_H_INCLUDE_
