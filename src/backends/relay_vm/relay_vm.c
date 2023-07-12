/**
 * @file relay_vm/relay_vm.c
 * @brief The implementation for relay_vm public api.
 */

#include <device/cpu_memory.h>
#include <relay_vm/relay_vm.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

#define CHECK_RelayVirtualMachine(vm) CHECK_INPUT_POINTER(vm, -2, "RelayVirtualMachine")
#define CHECK_DeviceList(_devices, _num_dev)                                                       \
    do {                                                                                           \
        CHECK_INPUT_POINTER((_devices), NULL, "Devices");                                          \
        if (unlikely((_num_dev) == 0)) {                                                           \
            TVM_RT_SET_ERROR_RETURN(                                                               \
                NULL, "Invalid argument: the number of devices cannot be zero, at least 1.");      \
        }                                                                                          \
    } while (0)

static TVM_RT_WASM_RelayVirtualMachine
TVM_RT_WASM_RelayVirtualMachineCreateFromExecutable(TVM_RT_WASM_RelayExecutable exec) {

    TVM_RT_WASM_RelayVirtualMachine vm =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_RelayVirtualMachine_st));
    memset(vm, 0, sizeof(struct TVM_RT_WASM_RelayVirtualMachine_st));
    vm->exec = exec;

    size_t num_functions = exec->num_functions;
    vm->functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_RelayVMFunc_st) * num_functions);
    memset(vm->functions, 0, sizeof(struct TVM_RT_WASM_RelayVMFunc_st) * num_functions);
    TVM_RT_WASM_TrieCreate(&vm->function_map);
    for (size_t i = 0; i < num_functions; ++i) {
        TVM_RT_WASM_RelayVMFunc func = vm->functions + i;
        func->exec_func = exec->functions[i];
        size_t num_params = func->exec_func->num_params;
        // build param names map
        TVM_RT_WASM_TrieCreate(&func->params);
        for (size_t j = 0; j < num_params; ++j) {
            TVM_RT_WASM_TrieInsert(func->params, (const uint8_t *)func->exec_func->param_names[j],
                                   (void *)(uintptr_t)j);
        }
        func->inputs =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVM_RT_WASM_RelayVMRegister) * num_params);
        memset(func->inputs, 0, sizeof(TVM_RT_WASM_RelayVMRegister) * num_params);
        // inset the vm function
        TVM_RT_WASM_TrieInsert(vm->function_map, (const uint8_t *)func->exec_func->name,
                               (void *)func);
    }
    return vm;
}

TVM_RT_WASM_RelayVirtualMachine TVM_RT_WASM_RelayVirtualMachineCreate(TVMModuleHandle module_handle,
                                                                      const char *byte_code,
                                                                      uint32_t byte_code_size,
                                                                      const DLDevice *devices,
                                                                      uint32_t num_dev) {
    CHECK_DeviceList(devices, num_dev);
    TVM_RT_WASM_RelayExecutable exec = NULL;
    int status;
    StreamReader *reader;

    status = TVM_RT_WASM_BytesStreamReaderCreate(byte_code, byte_code_size, &reader);
    if (unlikely(status)) {
        return NULL;
    }

    status =
        TVM_RT_WASM_RelayExecutableCreateFromReader(module_handle, reader, devices, num_dev, &exec);
    reader->Free(reader);
    if (unlikely(status)) {
        return NULL;
    }
    return TVM_RT_WASM_RelayVirtualMachineCreateFromExecutable(exec);
}

TVM_RT_WASM_RelayVirtualMachine
TVM_RT_WASM_RelayVirtualMachineCreateFromFile(TVMModuleHandle module_handle, const char *filename,
                                              const DLDevice *devices, uint32_t num_dev) {
    CHECK_DeviceList(devices, num_dev);
    TVM_RT_WASM_RelayExecutable exec = NULL;
    int status;
    StreamReader *reader;

    status = TVM_RT_WASM_FileStreamReaderCreate(filename, &reader);
    if (unlikely(status)) {
        return NULL;
    }

    status =
        TVM_RT_WASM_RelayExecutableCreateFromReader(module_handle, reader, devices, num_dev, &exec);
    reader->Free(reader);
    if (unlikely(status)) {
        return NULL;
    }
    return TVM_RT_WASM_RelayVirtualMachineCreateFromExecutable(exec);
}

int TVM_RT_WASM_RelayVirtualMachineFree(TVM_RT_WASM_RelayVirtualMachine vm) {
    CHECK_RelayVirtualMachine(vm);
    TVM_RT_WASM_RelayVMRegisterFree(&vm->ret_register);
    // the current frame size should be zero
    if (vm->frames) {
        TVM_RT_WASM_HeapMemoryFree(vm->frames);
    }
    if (vm->functions) {
        for (size_t func_id = 0; func_id < vm->exec->num_functions; ++func_id) {
            TVM_RT_WASM_RelayVMFunc func = vm->functions + func_id;
            if (func->params) {
                TVM_RT_WASM_TrieRelease(func->params);
            }
            if (func->inputs) {
                for (size_t i = 0; i < func->exec_func->num_params; ++i) {
                    TVM_RT_WASM_RelayVMRegisterFree(func->inputs + i);
                }
                TVM_RT_WASM_HeapMemoryFree(func->inputs);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(vm->functions);
    }
    if (vm->function_map) {
        TVM_RT_WASM_TrieRelease(vm->function_map);
    }
    if (vm->exec) {
        TVM_RT_WASM_RelayExecutableFree(vm->exec);
        vm->exec = NULL;
    }
    TVM_RT_WASM_HeapMemoryFree(vm);
    return 0;
}

#define TVM_RT_WASM_RelayVMGetAndCheckFunc(_vm, _func_name)                                        \
    TVM_RT_WASM_RelayVMFunc func;                                                                  \
    do {                                                                                           \
        if ((_func_name) == NULL) {                                                                \
            (_func_name) = TVM_RT_WASM_RelayDefaultFunctionName;                                   \
        }                                                                                          \
        if (unlikely(TVM_RT_WASM_TrieQuery((_vm)->function_map, (const uint8_t *)(_func_name),     \
                                           (void **)&func) != TRIE_SUCCESS)) {                     \
            TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function `%s`", _func_name);                  \
        }                                                                                          \
    } while (0)

int TVM_RT_WASM_RelayVirtualMachineRun(TVM_RT_WASM_RelayVirtualMachine vm, const char *func_name) {
    CHECK_RelayVirtualMachine(vm);
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);
    // the frame size will keep zero.
    return TVM_RT_WASM_RelayVMRunFunction(vm, func);
}

static int TVM_RT_WASM_RelayVMSetInputInner(TVM_RT_WASM_RelayVirtualMachine vm,
                                            TVM_RT_WASM_RelayVMFunc func, uint32_t index,
                                            const DLTensor *data_in) {
    DLDevice input_dev = vm->exec->devices[func->exec_func->param_device_indices[index]];
    TVM_RT_WASM_RelayVMRegister *input_reg = func->inputs + index;
    if (data_in->device.device_type == input_dev.device_type &&
        data_in->device.device_id == input_dev.device_id) {
        input_reg->tp = Reg_BorrowedTensor;
        input_reg->tensor = *data_in;
    } else {
        uint64_t need_bytes =
            TVM_RT_WASM_DLTensor_GetDataBytes(data_in->shape, data_in->ndim, data_in->dtype);
        int should_alloc_data = 0;
        if (input_reg->tp == Reg_OwnedTensor) {
            uint64_t current_bytes = TVM_RT_WASM_DLTensor_GetDataBytes(
                input_reg->tensor.shape, input_reg->tensor.ndim, input_reg->tensor.dtype);
            if (unlikely(current_bytes != need_bytes)) {
                TVM_RT_WASM_RelayVMRegisterFree(input_reg);
                should_alloc_data = 1;
            }
        } else {
            TVM_RT_WASM_RelayVMRegisterFree(input_reg);
            should_alloc_data = 1;
        }

        input_reg->tp = Reg_OwnedTensor;
        input_reg->tensor.device = input_dev;
        input_reg->tensor.dtype = data_in->dtype;
        if (should_alloc_data) {
            int status = TVMDeviceAllocDataSpace(input_dev, need_bytes, 0, data_in->dtype,
                                                 &input_reg->tensor.data);
            if (unlikely(status)) {
                return status;
            }
            input_reg->tensor.ndim = data_in->ndim;
            input_reg->tensor.shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * data_in->ndim);
            memcpy(input_reg->tensor.shape, data_in->shape, sizeof(int64_t) * data_in->ndim);
        } else {
            // check if shapes need realloc
            if (input_reg->tensor.ndim != data_in->ndim) {
                input_reg->tensor.ndim = data_in->ndim;
                TVM_RT_WASM_HeapMemoryFree(input_reg->tensor.shape);
                input_reg->tensor.shape =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * data_in->ndim);
            }
            memcpy(input_reg->tensor.shape, data_in->shape, sizeof(int64_t) * data_in->ndim);
        }

        int status = TVMDeviceCopyDataFromTo((DLTensor *)data_in, &input_reg->tensor, NULL);
        if (unlikely(status)) {
            return status;
        }
    }
    return 0;
}

int TVM_RT_WASM_RelayVirtualMachineSetInput(TVM_RT_WASM_RelayVirtualMachine vm,
                                            const char *func_name, uint32_t index,
                                            const DLTensor *data_in) {
    CHECK_RelayVirtualMachine(vm);
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);
    CHECK_INDEX_RANGE((uint32_t)func->exec_func->num_params, index);
    return TVM_RT_WASM_RelayVMSetInputInner(vm, func, index, data_in);
}

int TVM_RT_WASM_RelayVirtualMachineSetInputByName(TVM_RT_WASM_RelayVirtualMachine vm,
                                                  const char *func_name, const char *name,
                                                  const DLTensor *data_in) {
    CHECK_RelayVirtualMachine(vm);
    CHECK_INPUT_POINTER(name, -2, "Name");
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);

    uintptr_t index;
    if (unlikely(TVM_RT_WASM_TrieQuery(func->params, (const uint8_t *)name, (void **)&index) !=
                 TRIE_SUCCESS)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function param `%s`", name);
    }
    return TVM_RT_WASM_RelayVMSetInputInner(vm, func, (uint32_t)index, data_in);
}

int TVM_RT_WASM_RelayVirtualMachineGetOutput(TVM_RT_WASM_RelayVirtualMachine vm,
                                             const char *func_name, uint32_t index,
                                             DLTensor *data_out) {
    CHECK_RelayVirtualMachine(vm);
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);

    switch (vm->ret_register.tp) {
    case Reg_OwnedTensor:
    case Reg_BorrowedTensor:
        CHECK_INDEX_RANGE(1, index);
        return TVMDeviceCopyDataFromTo(&vm->ret_register.tensor, data_out, NULL);
    default:
        TVM_RT_SET_ERROR_RETURN(-1, "No output now.");
    }
}

/*---------------Functions to get relay virtual machine information-------------------------------*/

int TVM_RT_WASM_RelayVirtualMachineGetInputIndex(TVM_RT_WASM_RelayVirtualMachine vm,
                                                 const char *func_name, const char *name) {
    CHECK_RelayVirtualMachine(vm);
    CHECK_INPUT_POINTER(name, -2, "Name");
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);

    uintptr_t index;
    if (unlikely(TVM_RT_WASM_TrieQuery(func->params, (const uint8_t *)name, (void **)&index) !=
                 TRIE_SUCCESS)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function param `%s`", name);
    }
    return (int)index;
}

int TVM_RT_WASM_RelayVirtualMachineGetNumInputs(TVM_RT_WASM_RelayVirtualMachine vm,
                                                const char *func_name) {
    CHECK_RelayVirtualMachine(vm);
    TVM_RT_WASM_RelayVMGetAndCheckFunc(vm, func_name);
    return (int)func->exec_func->num_params;
}

int TVM_RT_WASM_RelayVirtualMachineGetNumOutputs(TVM_RT_WASM_RelayVirtualMachine vm) {
    CHECK_RelayVirtualMachine(vm);

    // todo: adt
    switch (vm->ret_register.tp) {
    case Reg_OwnedTensor:
    case Reg_BorrowedTensor:
        return 1;
    default:
        return 0;
    }
}
