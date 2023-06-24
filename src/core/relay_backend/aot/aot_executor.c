#include <aot_executor.h>
#include <module/module.h>
#include <relay_backend/aot/aot_executor.h>
#include <string.h>
#include <utils/tensor_helper.h>

_Pragma(TOSTRING(weak TVM_GET_METADATA_FUNC)) int32_t
    TVM_GET_METADATA_FUNC(TVMValue *arg_values, const int *arg_type_codes, int num_args, const TVMValue *ret_values,
                          const int *ret_type_codes, const void *resource_handle) {
    (void)arg_values;
    (void)arg_type_codes;
    (void)ret_values;
    (void)ret_type_codes;
    (void)resource_handle;
    return num_args - 1;
}

static int TVM_RT_WASM_AotExecutorAllocStorage(TVM_RT_WASM_AotExecutor a) {
    int status = 0;
    const struct TVMMetadata *metadata = a->metadata;
    const DLDevice device = a->devices[0];

    // alloc tensors
    size_t args_size = (size_t)(metadata->num_inputs + metadata->num_outputs);
    if (metadata->num_workspace_pools > 0) {
        args_size += metadata->num_workspace_pools + (metadata->num_constant_pools ? 1 : 0);
    }
    size_t tensor_bytes_size = sizeof(DLTensor) * args_size;
    a->tensors = TVM_RT_WASM_HeapMemoryAlloc(tensor_bytes_size);
    memset(a->tensors, 0, tensor_bytes_size);
    DLTensor *now_tensor = a->tensors;
    // input tensors
    for (int64_t i = 0; i < metadata->num_inputs; ++i) {
        now_tensor->shape = (int64_t *)metadata->inputs[i].shape;
        now_tensor->dtype = metadata->inputs[i].dtype;
        now_tensor->ndim = (int32_t)metadata->inputs[i].num_shape;
        now_tensor->device = device;
        size_t data_bytes = TVM_RT_WASM_DLTensor_GetDataBytes(now_tensor);
        status = TVMDeviceAllocDataSpace(device, data_bytes, 0, now_tensor->dtype, &now_tensor->data);
        if (unlikely(status)) {
            return -1;
        }
        ++now_tensor;
    }
    // output tensors
    for (int64_t i = 0; i < metadata->num_outputs; ++i) {
        now_tensor->shape = (int64_t *)metadata->outputs[i].shape;
        now_tensor->dtype = metadata->outputs[i].dtype;
        now_tensor->ndim = (int32_t)metadata->outputs[i].num_shape;
        now_tensor->device = device;
        size_t data_bytes = TVM_RT_WASM_DLTensor_GetDataBytes(now_tensor);
        status = TVMDeviceAllocDataSpace(device, data_bytes, 0, now_tensor->dtype, &now_tensor->data);
        if (unlikely(status)) {
            return -1;
        }
        ++now_tensor;
    }
    if (metadata->num_workspace_pools > 0) {
        // constant pool
        if (metadata->num_constant_pools) {
            // todo
            TVM_RT_NOT_IMPLEMENT(-1);
        }

        // workspace pool
        for (int64_t i = 0; i < metadata->num_workspace_pools; ++i) {
            now_tensor->shape = (int64_t *)metadata->workspace_pools[i].shape;
            now_tensor->dtype = metadata->workspace_pools[i].dtype;
            now_tensor->ndim = (int32_t)metadata->workspace_pools[i].num_shape;
            now_tensor->device = device;
            size_t data_bytes = TVM_RT_WASM_DLTensor_GetDataBytes(now_tensor);
            status = TVMDeviceAllocDataSpace(device, data_bytes, 0, now_tensor->dtype, &now_tensor->data);
            if (unlikely(status)) {
                return -1;
            }
            ++now_tensor;
        }
    }

    a->tvm_args_value = TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMValue) * args_size);
    a->tvm_args_type = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int) * args_size);
    for (size_t i = 0; i < args_size; ++i) {
        a->tvm_args_value[i].v_handle = a->tensors + i;
    }
    for (size_t i = 0; i < args_size; ++i) {
        a->tvm_args_type[i] = kTVMDLTensorHandle;
    }
    a->tvm_args_size = (int)args_size;
    return status;
}

TVM_RT_WASM_AotExecutor TVM_RT_WASM_AotExecutorCreate(TVMModuleHandle module_handle, const DLDevice *devices,
                                                      uint32_t num_dev) {
    // if module_handle is NULL, use the system library.
    if (module_handle == NULL) {
        SET_TIME(t0)
        int status = TVM_RT_WASM_ModuleFactory(MODULE_SYSTEM_LIB, NULL, 0, (Module **)&module_handle);
        if (unlikely(status)) {
            return NULL;
        }
        SET_TIME(t1)
        DURING_PRINT(t1, t0, "sys_lib_create time");
    }
    CHECK_INPUT_POINTER(devices, NULL, "Devices");
    if (unlikely(num_dev == 0)) {
        TVM_RT_SET_ERROR_RETURN(NULL, "Invalid argument: the number of devices cannot be zero, at least 1.");
    }

    Module *mod = (Module *)module_handle;
    TVMValue ret_value = {.v_handle = NULL};
    int ret_type_code = 0;
    // try to use system library
    int status = TVM_GET_METADATA_FUNC(NULL, NULL, 0, &ret_value, &ret_type_code, NULL);
    if (status == -1) { // no `get_c_metadata` function in system library.

        // try to get `get_metadata` function.
        PackedFunction *get_metadata_func;
        status = mod->GetFunction(mod, TVM_GET_METADATA_FUNC_NAME, 1, (TVMFunctionHandle *)&get_metadata_func);
        if (unlikely(status)) {
            TVM_RT_SET_ERROR_RETURN(NULL, "Cannot find function `%s`.", TVM_GET_METADATA_FUNC_NAME);
        }

        // call the function to get metadata
        get_metadata_func->exec(NULL, NULL, 0, &ret_value, &ret_type_code, NULL);
    }

    if (ret_type_code != kTVMOpaqueHandle) {
        TVM_RT_SET_ERROR_RETURN(NULL, "`%s` should return kTVMOpaqueHandle but got %d", TVM_GET_METADATA_FUNC_NAME,
                                ret_type_code);
    }
    struct TVMMetadata *metadata = (struct TVMMetadata *)ret_value.v_handle;
    if (unlikely(!metadata)) {
        TVM_RT_SET_ERROR_RETURN(NULL, "`%s` cannot return NULL", TVM_GET_METADATA_FUNC_NAME);
    }

    TVM_RT_WASM_AotExecutor a = TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_AotExecutor_st));
    memset(a, 0, sizeof(struct TVM_RT_WASM_AotExecutor_st));
    //    a->module_handle = module_handle;
    a->metadata = metadata;

    // find main function
    size_t main_func_name_len = strlen(metadata->mod_name) + sizeof(TVM_MODULE_MAIN) + 1;
    char *main_func_name = TVM_RT_WASM_WorkplaceMemoryAlloc(main_func_name_len);
    strcpy(main_func_name, metadata->mod_name);
    strcat(main_func_name, "_" TVM_MODULE_MAIN);
    status = mod->GetFunction(mod, main_func_name, 1, (TVMFunctionHandle *)&a->main_func);
    if (unlikely(status)) {
        TVM_RT_SET_ERROR("Cannot find function `%s`.", main_func_name);
        TVM_RT_WASM_WorkplaceMemoryFree(main_func_name);
        TVM_RT_WASM_AotExecutorFree(a);
        return NULL;
    } else {
        TVM_RT_WASM_WorkplaceMemoryFree(main_func_name);
    }

    // copy devices
    DLDevice *devs = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDevice) * num_dev);
    memcpy(devs, devices, sizeof(DLDevice) * num_dev);
    a->devices = devs;
    //    a->num_devices = num_dev;

    status = TVM_RT_WASM_AotExecutorAllocStorage(a);
    if (unlikely(status)) {
        TVM_RT_WASM_AotExecutorFree(a);
        return NULL;
    }
    return a;
}

int TVM_RT_WASM_AotExecutorFree(TVM_RT_WASM_AotExecutor a) {
    CHECK_AotExecutor(a);
    if (a->devices) {
        TVM_RT_WASM_HeapMemoryFree(a->devices);
    }
    if (a->tvm_args_value) {
        TVM_RT_WASM_HeapMemoryFree(a->tvm_args_value);
    }
    if (a->tvm_args_type) {
        TVM_RT_WASM_HeapMemoryFree(a->tvm_args_type);
    }
    if (a->tensors) {
        for (int64_t i = 0; i < (a->metadata->num_inputs + a->metadata->num_outputs); ++i) {
            if (a->tensors[i].data) {
                TVMDeviceFreeDataSpace(a->tensors[i].device, a->tensors[i].data);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(a->tensors);
    }
    TVM_RT_WASM_HeapMemoryFree(a);
    return 0;
}

int TVM_RT_WASM_AotExecutorRun(TVM_RT_WASM_AotExecutor a) {
    CHECK_AotExecutor(a);
    TVMValue ret_value;
    int ret_type_code;
    int status =
        a->main_func->exec(a->tvm_args_value, a->tvm_args_type, a->tvm_args_size, &ret_value, &ret_type_code, NULL);
    return status;
}

int TVM_RT_WASM_AotExecutorSetInput(TVM_RT_WASM_AotExecutor a, uint32_t index, const DLTensor *data_in) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    CHECK_NodeRange((uint32_t)a->metadata->num_inputs, index);
    return TVMDeviceCopyDataFromTo((DLTensor *)data_in, &a->tensors[index], NULL);
}

int TVM_RT_WASM_AotExecutorSetInputByName(TVM_RT_WASM_AotExecutor a, const char *name, const DLTensor *data_in) {
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    int index = TVM_RT_WASM_AotExecutorGetInputIndex(a, name);
    if (unlikely(index == -1)) {
        return index;
    }
    return TVM_RT_WASM_AotExecutorSetInput(a, index, data_in);
}

int TVM_RT_WASM_AotExecutorGetOutput(TVM_RT_WASM_AotExecutor a, uint32_t index, DLTensor *data_out) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    CHECK_NodeRange((uint32_t)a->metadata->num_outputs, index);
    return TVMDeviceCopyDataFromTo(&a->tensors[a->metadata->num_inputs + index], data_out, NULL);
}

int TVM_RT_WASM_AotExecutorGetOutputByName(TVM_RT_WASM_AotExecutor a, const char *name, DLTensor *data_out) {
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    int index = TVM_RT_WASM_AotExecutorGetOutputIndex(a, name);
    if (unlikely(index == -1)) {
        return index;
    }
    return TVM_RT_WASM_AotExecutorGetOutput(a, index, data_out);
}

/*-------------------------Functions to get AotExecutor information---------------------------------------------------*/

int TVM_RT_WASM_AotExecutorGetInputIndex(TVM_RT_WASM_AotExecutor a, const char *name) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(name, -2, "Name");

    for (int64_t i = 0; i < a->metadata->num_inputs; ++i) {
        if (!strcmp(a->metadata->inputs[i].name, name)) {
            return (int)i;
        }
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Node name `%s` not found", name);
}

int TVM_RT_WASM_AotExecutorGetOutputIndex(TVM_RT_WASM_AotExecutor a, const char *name) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(name, -2, "Name");

    for (int64_t i = 0; i < a->metadata->num_outputs; ++i) {
        if (!strcmp(a->metadata->outputs[i].name, name)) {
            return (int)i;
        }
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Node name `%s` not found", name);
}

int TVM_RT_WASM_AotExecutorGetNumInputs(TVM_RT_WASM_AotExecutor a) {
    CHECK_AotExecutor(a);
    return (int)a->metadata->num_inputs;
}

int TVM_RT_WASM_AotExecutorGetNumOutputs(TVM_RT_WASM_AotExecutor a) {
    CHECK_AotExecutor(a);
    return (int)a->metadata->num_outputs;
}

int TVM_RT_WASM_AotExecutorGetInputDataType(TVM_RT_WASM_AotExecutor a, uint32_t index, DLDataType *type_ptr) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(type_ptr, -2, "DLDataType pointer");
    CHECK_NodeRange((uint32_t)a->metadata->num_inputs, index);

    *type_ptr = a->metadata->inputs[index].dtype;
    return 0;
}

int TVM_RT_WASM_AotExecutorGetOutputDataType(TVM_RT_WASM_AotExecutor a, uint32_t index, DLDataType *type_ptr) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(type_ptr, -2, "DLDataType pointer");
    CHECK_NodeRange((uint32_t)a->metadata->num_outputs, index);

    *type_ptr = a->metadata->outputs[index].dtype;
    return 0;
}

int TVM_RT_WASM_AotExecutorGetInputShape(TVM_RT_WASM_AotExecutor a, uint32_t index, const int64_t **shape_ptr,
                                         int32_t *ndim_ptr) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(shape_ptr, -2, "shape pointer");
    CHECK_INPUT_POINTER(ndim_ptr, -2, "ndim pointer");
    CHECK_NodeRange((uint32_t)a->metadata->num_inputs, index);

    const struct TVMTensorInfo *in_tensor = a->metadata->inputs + index;
    *shape_ptr = in_tensor->shape;
    *ndim_ptr = (int32_t)in_tensor->num_shape;
    return 0;
}

int TVM_RT_WASM_AotExecutorGetOutputShape(TVM_RT_WASM_AotExecutor a, uint32_t index, const int64_t **shape_ptr,
                                          int32_t *ndim_ptr) {
    CHECK_AotExecutor(a);
    CHECK_INPUT_POINTER(shape_ptr, -2, "shape pointer");
    CHECK_INPUT_POINTER(ndim_ptr, -2, "ndim pointer");
    CHECK_NodeRange((uint32_t)a->metadata->num_outputs, index);

    const struct TVMTensorInfo *out_tensor = a->metadata->outputs + index;
    *shape_ptr = out_tensor->shape;
    *ndim_ptr = (int32_t)out_tensor->num_shape;
    return 0;
}
