/**
 * @file relay_vm/relay_vm.h
 * @brief Private struct and functions for relay_vm.
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <relay_vm.h>
#include <relay_vm/relay_executable.h>
#include <relay_vm/relay_vm_register.h>
#include <utils/trie.h>

/** @brief Relay Virtual Machine Function Frame. */
typedef struct TVM_RT_WASM_RelayVMFrame_st {
    /** @brief current code section */
    const TVM_RT_WASM_RelayInstruction *code;

    /** @brief registers in this frame */
    TVM_RT_WASM_RelayVMRegister *registers;

    /** @brief registers size in this frame */
    size_t num_registers;

    /** @brief current program counter */
    size_t pc;

    /** @brief Register in caller's frame to put return value */
    size_t caller_return_register;
} TVM_RT_WASM_RelayVMFrame;

/** @brief Relay Virtual Machine Functions information */
typedef struct TVM_RT_WASM_RelayVMFunc_st {
    TVM_RT_WASM_RelayFunction exec_func;

    /** @brief map <param_name, param_index> */
    Trie *params;

    /** @brief the function input values */
    TVM_RT_WASM_RelayVMRegister *inputs;
} *TVM_RT_WASM_RelayVMFunc;

/** @brief The Relay Virtual Machine */
struct TVM_RT_WASM_RelayVirtualMachine_st {
    /** @brief The TVM relay executable */
    TVM_RT_WASM_RelayExecutable exec;

    /** @brief map <function_name, TVM_RT_WASM_RelayVMFunc> */
    Trie *function_map;
    /** @brief function list */
    struct TVM_RT_WASM_RelayVMFunc_st *functions;

    /** @brief frame stack */
    struct TVM_RT_WASM_RelayVMFrame_st *frames;
    size_t frame_stack_size;
    size_t frame_stack_capacity;

    /** @brief current frame */
    TVM_RT_WASM_RelayVMFrame current_frame;

    /** @brief special register, save the return value for called function */
    TVM_RT_WASM_RelayVMRegister ret_register;
};

/**
 * @brief Run the given function for relay vm.
 * @param vm The relay vm instance.
 * @param func The relay vm function instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayVMRunFunction(TVM_RT_WASM_RelayVirtualMachine vm,
                                   TVM_RT_WASM_RelayVMFunc func);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_H_INCLUDE_
