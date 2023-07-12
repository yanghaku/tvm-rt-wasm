/**
 * @file relax_vm.h
 * @brief Interfaces for relax virtual machine.
 */

#ifndef TVM_RT_WASM_RELAX_VM_H_INCLUDE_
#define TVM_RT_WASM_RELAX_VM_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

#define TVM_RT_WASM_RelaxDefaultFunctionName "main"

typedef struct TVM_RT_WASM_RelaxVirtualMachine_st *TVM_RT_WASM_RelaxVirtualMachine;

/**
 * @brief Allocate a new TVM_RT_WASM_RelaxVirtualMachine and initialize it.
 * @param module_handle TVM relax executable library module. If NULL, use the system library.
 * @param devices runtime execution device.
 * @param num_dev the number of devices.
 * @note The function will get ownership of this module_handle if create successfully.
 *       **DO NOT** free this module_handle if create successfully!
 * @return Pointer of TVM_RT_WASM_RelaxVirtualMachine instance if successful, NULL if fail.
 */
TVM_DLL TVM_RT_WASM_RelaxVirtualMachine TVM_RT_WASM_RelaxVirtualMachineCreate(
    TVMModuleHandle module_handle, const DLDevice *devices, uint32_t num_dev);

/**
 * @brief Free the instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineFree(TVM_RT_WASM_RelaxVirtualMachine vm);

/**
 * @brief Execute the VM function.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineRun(TVM_RT_WASM_RelaxVirtualMachine vm,
                                               const char *func_name);

/**
 * @brief Set input to the vm based on index.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index the index of inputs.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineSetInput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                    const char *func_name, uint32_t index,
                                                    const DLTensor *data_in);

/**
 * @brief Set input to the vm based on name.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name the name string for node.
 * @param data_in The input data.
 * @note If the device is same, the function will zero copy.
 *       else, the function will copy `data_in` to vm input tensor.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineSetInputByName(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                          const char *func_name, const char *name,
                                                          const DLTensor *data_in);

/**
 * @brief Get output data for given output index.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param index The output index.
 * @param data_out The point to DLTensor. The function will copy vm output tensor to `data_out`.
 * @return 0 if successful.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetOutput(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                     const char *func_name, uint32_t index,
                                                     DLTensor *data_out);

/*-----------------Functions to get relax virtual machine information-----------------------------*/

/**
 * @brief Get the input index given the name of input.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @param name The name of the input.
 * @return The index of input. If cannot find name or error, return -1.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetInputIndex(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                         const char *func_name, const char *name);

/**
 * @brief Get number of input tensors allocated.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return integer number of input tensors.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetNumInputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                        const char *func_name);

/**
 * @brief Get number of output of current relax VM.
 * @param vm The instance of TVM_RT_WASM_RelaxVirtualMachine.
 * @param func_name The function name. if func_name is NULL, use the default name "main".
 * @return integer number of output tensors.
 */
TVM_DLL int TVM_RT_WASM_RelaxVirtualMachineGetNumOutputs(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                         const char *func_name);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_RELAX_VM_H_INCLUDE_
