/**
 * @file aot_executor.h
 * @brief Interfaces for aot executor.
 */

#ifndef TVM_RT_WASM_AOT_EXECUTOR_H_INCLUDE_
#define TVM_RT_WASM_AOT_EXECUTOR_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

typedef struct TVM_RT_WASM_AotExecutor_st *TVM_RT_WASM_AotExecutor;

/**
 * @brief Allocate a new TVM_RT_WASM_AotExecutor and initialize it.
 *
 * @param module_handle TVM aot executor library module. If NULL, use the system library.
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @return Pointer of TVM_RT_WASM_AotExecutor instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_AotExecutor TVM_RT_WASM_AotExecutorCreate(TVMModuleHandle module_handle,
                                                              const DLDevice *devices,
                                                              uint32_t num_dev);

/**
 * @brief Free the instance of TVM_RT_WASM_AotExecutor.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorFree(TVM_RT_WASM_AotExecutor a);

/**
 * @brief Execute the AOT module.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorRun(TVM_RT_WASM_AotExecutor a);

/**
 * @brief Set input to the Aot based on index.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index the index of inputs.
 * @param data_in The input data. The function will copy `data_in` to Aot input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorSetInput(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                            const DLTensor *data_in);

/**
 * @brief Set input to the Aot based on name.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param name the name string for node.
 * @param data_in The input data. The function will copy `data_in` to Aot input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorSetInputByName(TVM_RT_WASM_AotExecutor a, const char *name,
                                                  const DLTensor *data_in);

/**
 * @brief Get output data for given output index.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index The output index.
 * @param data_out The point to DLTensor. The function will copy Aot output tensor to `data_out`.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetOutput(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                             DLTensor *data_out);

/**
 * @brief Get output data for given output name.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param name The name of the output.
 * @param data_out The point to DLTensor. The function will copy Aot output tensor to `data_out`.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetOutputByName(TVM_RT_WASM_AotExecutor a, const char *name,
                                                   DLTensor *data_out);

/*-----------------Functions to get AotExecutor information---------------------------------------*/

/**
 * @brief Get the input index given the name of input.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param name The name of the input.
 * @return The index of input. If cannot find name or error, return -1.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetInputIndex(TVM_RT_WASM_AotExecutor a, const char *name);

/**
 * @brief Get the output index given the name of output.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param name The name of the output.
 * @return The index of input. If cannot find name, return -1.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetOutputIndex(TVM_RT_WASM_AotExecutor a, const char *name);

/**
 * @brief Get number of input tensors allocated.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @return integer number of input tensors.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetNumInputs(TVM_RT_WASM_AotExecutor a);

/**
 * @brief Get number of output tensors allocated.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @return integer number of output tensors.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetNumOutputs(TVM_RT_WASM_AotExecutor a);

/**
 * @brief Get the input DLTensor's data type.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index The index of inputs.
 * @param type_ptr The pointer to receive data type.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetInputDataType(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                                    DLDataType *type_ptr);

/**
 * @brief Get the output DLTensor's data type.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index The index of outputs.
 * @param type_ptr The pointer to receive data type.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetOutputDataType(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                                     DLDataType *type_ptr);

/**
 * @brief Get the input DLTensor's data shape.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index The index of inputs.
 * @param shape_ptr The pointer to receive shape array.
 * @param ndim_ptr The pointer to receive shape array length.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetInputShape(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                                 const int64_t **shape_ptr, int32_t *ndim_ptr);

/**
 * @brief Get the output DLTensor's data shape.
 * @param a The instance of TVM_RT_WASM_AotExecutor.
 * @param index The index of output.
 * @param shape_ptr The pointer to receive shape array.
 * @param ndim_ptr The pointer to receive shape array length.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_AotExecutorGetOutputShape(TVM_RT_WASM_AotExecutor a, uint32_t index,
                                                  const int64_t **shape_ptr, int32_t *ndim_ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_AOT_EXECUTOR_H_INCLUDE_
