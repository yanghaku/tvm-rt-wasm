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
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle,
                                            const DLDevice *devices, uint32_t num_dev, TVM_RT_WASM_GraphExecutor *g);

/*!
 * \brief Destory the instance of TVM_RT_WASM_GraphExecutor.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return 0 if successful
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorDestory(TVM_RT_WASM_GraphExecutor *g);

/*!
 * \brief Get total number of nodes.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return Total number of nodes.
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumOfNodes(const TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get the name of node for given index.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param nid the node index
 * \param name the pointer to receive string pointer
 * \return 0 if successful
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetNodeName(const TVM_RT_WASM_GraphExecutor g, uint32_t nid,
                                                        const char **name);

/*!
 * \brief Get the input index given the name of input.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name The name of the input.
 * \return The index of input.
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetInputIndex(const TVM_RT_WASM_GraphExecutor g, const char *name);

/*!
 * \brief Get the output index given the name of output.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name The name of the output.
 * \return The index of output.
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutputIndex(const TVM_RT_WASM_GraphExecutor g, const char *name);

/*!
 * \brief Get number of input tensors allocated.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return integer number of tensors available to use.
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumInputs(const TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get number of output tensors allocated.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return integer number of output tensors allocated.
 */
inline TVM_DLL int TVM_RT_WASM_GraphExecutorGetNumOutputs(const TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Get input to the graph based on name.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index the index of inputs.
 * \param data_in The input data.
 * \return 0 if successful
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorSetInput(TVM_RT_WASM_GraphExecutor g, uint32_t index, const DLTensor *data_in);

/*!
 * \brief Get input to the graph based on name.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param name the name string for node
 * \param data_in The input data.
 * \return 0 if successful
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorSetInputByName(TVM_RT_WASM_GraphExecutor g, const char *name,
                                                    const DLTensor *data_in);

/*!
 * \brief Get output data for given output index.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param index The output index.
 * \param data_out The DLTensor corresponding to given output node index.
 * \return The result of this function execution.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorGetOutput(TVM_RT_WASM_GraphExecutor g, uint32_t index, DLTensor *data_out);

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorLoadParams(TVM_RT_WASM_GraphExecutor g, const char *param_blob,
                                                uint32_t param_size);

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param filename File path to read and load
 * \return The result of this function execution.
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorLoadParamsFromFile(TVM_RT_WASM_GraphExecutor g, const char *filename);

/*!
 * \brief Execute the graph.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \return 0 if successful
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorRun(TVM_RT_WASM_GraphExecutor g);

/*!
 * \brief Clone a new instance of TVM_RT_WASM_GraphExecutor.
 * \param g The instance of TVM_RT_WASM_GraphExecutor.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
TVM_DLL int TVM_RT_WASM_GraphExecutorClone(TVM_RT_WASM_GraphExecutor g, TVM_RT_WASM_GraphExecutor *cloned);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_GRAPH_EXECUTOR_H
