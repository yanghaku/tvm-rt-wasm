/*!
 * \file graph_executor.h
 * \brief Interfaces for graph executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_GRAPH_EXECUTOR_H
#define TVM_RT_WASM_GRAPH_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

typedef struct TVM_RT_WASM_GraphExecutor_st *TVM_RT_WASM_GraphExecutor;

/*!
 * \brief Allocate a new TVM_RT_WASM_GraphExecutor and initialize it.
 *
 * \param graph_json JSON-encoded TVM graph.
 * \param module_handle TVM graph executor library module. If NULL, use the system library.
 * \param devices runtime execution device.
 * \param num_dev the number of devices.
 * \return Pointer of TVM_RT_WASM_GraphExecutor instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_GraphExecutor TVM_RT_WASM_GraphExecutorCreate(const char *graph_json,
                                                                  TVMModuleHandle module_handle,
                                                                  const DLDevice *devices,
                                                                  uint32_t num_dev);

/*!
 * \brief Free the instance of TVM_RT_WASM_GraphExecutor.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorFree(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Execute the graph.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorRun(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Set input to the graph based on index.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index the index of inputs.
 * \param data_in The input data. The function will copy `data_in` to graph input tensor.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorSetInput(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                              const DLTensor *data_in);

/*!
 * \brief Set input to the graph based on name.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name the name string for node.
 * \param data_in The input data. The function will copy `data_in` to graph input tensor.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorSetInputByName(TVM_RT_WASM_GraphExecutor g, const char *name,
                                                    const DLTensor *data_in);

/*!
 * \brief Get output data for given output index.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The output index.
 * \param data_out The point to DLTensor. The function will copy graph output tensor to `data_out`.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutput(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                               DLTensor *data_out);

/*!
 * \brief Get output data for given output name.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name The name of the output.
 * \param data_out The point to DLTensor. The function will copy graph output tensor to `data_out`.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutputByName(TVM_RT_WASM_GraphExecutor g, const char *name,
                                                     DLTensor *data_out);

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorLoadParams(TVM_RT_WASM_GraphExecutor g, const char *param_blob,
                                                uint32_t param_size);

/*!
 * \brief Load parameters from parameter file.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param filename File path to read and load.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorLoadParamsFromFile(TVM_RT_WASM_GraphExecutor g,
                                                        const char *filename);

/*!
 * \brief Clone a new instance of TVM_RT_WASM_GraphExecutor.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorClone(TVM_RT_WASM_GraphExecutor g,
                                           TVM_RT_WASM_GraphExecutor *cloned);

/*-----------------Functions to get graph information---------------------------------------------*/

/*!
 * \brief Get total number of nodes.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return Total number of nodes.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumOfNodes(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get the name of node for given index.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param nid the node index.
 * \param name the pointer to receive string pointer.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetNodeName(TVM_RT_WASM_GraphExecutor g, uint32_t nid,
                                                 const char **name);

/*!
 * \brief Get the input index given the name of input.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name The name of the input.
 * \return The index of input. If cannot find name or error, return -1.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetInputIndex(TVM_RT_WASM_GraphExecutor g, const char *name);

/*!
 * \brief Get the output index given the name of output.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name The name of the output.
 * \return The index of input. If cannot find name, return -1.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutputIndex(TVM_RT_WASM_GraphExecutor g, const char *name);

/*!
 * \brief Get number of input tensors allocated.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return integer number of input tensors.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumInputs(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get number of output tensors allocated.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return integer number of output tensors.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumOutputs(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get the input DLTensor's data type.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The index of inputs.
 * \param type_ptr The pointer to receive data type.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetInputDataType(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                                      DLDataType *type_ptr);

/*!
 * \brief Get the output DLTensor's data type.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The index of outputs.
 * \param type_ptr The pointer to receive data type.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutputDataType(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                                       DLDataType *type_ptr);

/*!
 * \brief Get the input DLTensor's data shape.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The index of inputs.
 * \param shape_ptr The pointer to receive shape array.
 * \param ndim_ptr The pointer to receive shape array length.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetInputShape(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                                   const int64_t **shape_ptr, int32_t *ndim_ptr);

/*!
 * \brief Get the output DLTensor's data shape.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The index of output.
 * \param shape_ptr The pointer to receive shape array.
 * \param ndim_ptr The pointer to receive shape array length.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutputShape(TVM_RT_WASM_GraphExecutor g, uint32_t index,
                                                    const int64_t **shape_ptr, int32_t *ndim_ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_GRAPH_EXECUTOR_H
