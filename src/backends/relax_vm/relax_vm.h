/**
 * @file relax_vm/relax_vm.h
 * @brief Private struct and functions for relax_vm.
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <relax_vm.h>
#include <relax_vm/relax_executable.h>
#include <utils/trie.h>

/** @brief The relax VM Register to save values.
 * The struct RelaxVMRegisterData contains a atomic integer to save reference.
 * RelaxVMRegister is a pointer to point to struct RelaxVMRegisterData.
 */
typedef struct RelaxVMRegisterData {
    int ref_num;
    TVMArgTypeCode typecode;
    TVMValue value;
} *RelaxVMRegister;

/** @brief The Relax Virtual Machine function frame */
typedef struct RelaxVMFrame {
    /** @brief The program counter to set after return. */
    RelaxVMIndex return_pc;
    /** @brief Save the return value to caller's frame registers. */
    RelaxVMRegisterName reg_caller_return;
    /** @brief The registers in this frame. */
    RelaxVMRegister *registers;
    size_t register_size;
    size_t register_capacity;
} RelaxVMFrame;

/** @brief The Relax VM Function state. save its inputs and outputs.
 * In the RelaxVMRegister array, the last is output, the rests are inputs.
 */
typedef struct RelaxVMFunctionInputsOutput {
    RelaxVMRegister *inputs_output;
    size_t num_inputs;
} RelaxVMFunctionInputsOutput;

/** @brief The Relax Virtual Machine */
struct TVM_RT_WASM_RelaxVirtualMachine_st {
    /** @brief The TVM relax executable */
    RelaxExecutableModule *exec_module;
    /** @brief The constant values in device to run. */
    RelaxVMRegister constants;
    /** @brief The buffer to save arguments to call packed functions. */
    TVMValue *call_packed_args_value;
    int *call_packed_args_typecode;
    /** @brief Devices to run this VM. */
    DLDevice *devices;
    size_t num_device;

    /** @brief The map <function name, RelaxVMFunctionInputsOutput>. */
    Trie *func_inputs_output_map;
    /** @brief The current program counter */
    RelaxVMIndex pc;
    /** @brief The current frame stack */
    RelaxVMFrame *frames;
    size_t frame_size;
    size_t frame_capacity;

    /** @brief The special register to pointer this relax VM. */
    struct RelaxVMRegisterData vm_register;
};

/** @brief Register the vm.builtin.* functions
 *  @return 0 if successful.
 */
int TVM_RT_WASM_RelaxVMRegisterBuiltinGlobalFunctions();

/**
 * @brief Run the Relax VM function.
 * @param vm The Relax VM instance.
 * @param func The Relax VM function.
 * @param inputs_output The input registers and output register.
 * @note All the inputs register values must not be null.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelaxVMRunFunction(TVM_RT_WASM_RelaxVirtualMachine vm, RelaxFunctionInfo *func,
                                   RelaxVMFunctionInputsOutput *inputs_output);

/** @brief Free the register value. */
#define TVM_RT_WASM_RelaxVMRegisterFree(_reg)                                                      \
    do {                                                                                           \
        if ((_reg) != NULL && (--(_reg)->ref_num) <= 0) {                                          \
            switch ((_reg)->typecode) {                                                            \
            case kTVMDLTensorHandle:                                                               \
            default:                                                                               \
                break;                                                                             \
            }                                                                                      \
            TVM_RT_WASM_HeapMemoryFree(_reg);                                                      \
        }                                                                                          \
    } while (0)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_H_INCLUDE_
