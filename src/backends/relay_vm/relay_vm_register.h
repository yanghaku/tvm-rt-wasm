/*!
 * @file relay_vm/relay_vm_register.h
 * @brief relay VM register struct.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_REGISTER_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_REGISTER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>

/** @brief Relay VM Register type */
enum TVM_RT_WASM_RelayVMRegisterType {
    Reg_Null = 0U,
    Reg_BorrowedTensor = 1U,
    Reg_OwnedTensor = 2U,
    Reg_Storage = 3U,
};

/** @brief Relay VM Register, save object data such as DLTensor */
typedef struct TVM_RT_WASM_RelayVMRegister_st {
    enum TVM_RT_WASM_RelayVMRegisterType tp;
    union {
        DLTensor tensor;
        struct {
            DLDevice device;
            void *data;
            size_t size;
            DLDataType dtype;
        } storage;
    };
} TVM_RT_WASM_RelayVMRegister;

INLINE void TVM_RT_WASM_RelayVMRegisterFree(TVM_RT_WASM_RelayVMRegister *reg) {
    switch (reg->tp) {
    case Reg_OwnedTensor:
        TVM_RT_WASM_HeapMemoryFree(reg->tensor.shape);
        TVMDeviceFreeDataSpace(reg->tensor.device, reg->tensor.data);
        break;
    default:
        break;
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_VM_REGISTER_H_INCLUDE_
