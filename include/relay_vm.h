/**
 * @file relay_vm.h
 * @brief Interfaces for relay virtual machine.
 * @sa https://tvm.apache.org/docs/arch/virtual_machine.html
 */

#ifndef TVM_RT_WASM_RELAY_VM_H_INCLUDE_
#define TVM_RT_WASM_RELAY_VM_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

#define TVM_RT_WASM_RelayDefaultFunctionName "main"

typedef struct TVM_RT_WASM_RelayVirtualMachine_st *TVM_RT_WASM_RelayVirtualMachine;

/**
 * @brief Allocate a new TVM_RT_WASM_RelayVirtualMachine and initialize it.
 * @param module_handle TVM relay executable library module. If NULL, use the system library.
 * @param byte_code TVM relay executable bytecode byte array.
 * @param byte_code_size TVM relay executable bytecode byte array length.
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @note The function will get ownership of this module_handle if create successfully.
 *       **DO NOT** free this module_handle if create successfully!
 * @return Pointer of TVM_RT_WASM_RelayVirtualMachine instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_RelayVirtualMachine TVM_RT_WASM_RelayVirtualMachineCreate(
    TVMModuleHandle module_handle, const char *byte_code, uint32_t byte_code_size,
    const DLDevice *devices, uint32_t num_dev);

/**
 * @brief Allocate a new TVM_RT_WASM_RelayVirtualMachine and initialize it.
 * @param module_handle TVM relay executable library module. If NULL, use the system library.
 * @param filename TVM relay executable bytecode file.
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @note The function will get ownership of this module_handle if create successfully.
 *       **DO NOT** free this module_handle if create successfully!
 * @return Pointer of TVM_RT_WASM_RelayVirtualMachine instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_RelayVirtualMachine TVM_RT_WASM_RelayVirtualMachineCreateFromFile(
    TVMModuleHandle module_handle, const char *filename, const DLDevice *devices, uint32_t num_dev);

/**
 * @brief Free the instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineFree(TVM_RT_WASM_RelayVirtualMachine vm);

/**
 * @brief Execute the VM function.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineRun(TVM_RT_WASM_RelayVirtualMachine vm,
                                               const char *func_name);

/**
 * @brief Set input to the vm based on index.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index the index of inputs.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineSetInput(TVM_RT_WASM_RelayVirtualMachine vm,
                                                    const char *func_name, uint32_t index,
                                                    const DLTensor *data_in);

/**
 * @brief Set input to the vm based on name.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name the name string for node.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineSetInputByName(TVM_RT_WASM_RelayVirtualMachine vm,
                                                          const char *func_name, const char *name,
                                                          const DLTensor *data_in);

/**
 * @brief Get output data for given output index.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index The output index.
 * @param data_out The point to DLTensor. The function will copy vm output tensor to `data_out`.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineGetOutput(TVM_RT_WASM_RelayVirtualMachine vm,
                                                     const char *func_name, uint32_t index,
                                                     DLTensor *data_out);

/*-----------------Functions to get relay virtual machine information-----------------------------*/

/**
 * @brief Get the input index given the name of input.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name The name of the input.
 * @return The index of input. If cannot find name or error, return -1.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineGetInputIndex(TVM_RT_WASM_RelayVirtualMachine vm,
                                                         const char *func_name, const char *name);

/**
 * @brief Get number of input tensors allocated.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return integer number of input tensors.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineGetNumInputs(TVM_RT_WASM_RelayVirtualMachine vm,
                                                        const char *func_name);

/**
 * @brief Get number of output of current relay VM.
 * @param vm The instance of TVM_RT_WASM_RelayVirtualMachine.
 * @return integer number of output tensors.
 */
TVM_DLL int TVM_RT_WASM_RelayVirtualMachineGetNumOutputs(TVM_RT_WASM_RelayVirtualMachine vm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_RELAY_VM_H_INCLUDE_
