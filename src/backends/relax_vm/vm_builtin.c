/**
 * @file relay_vm/vm_builtin.c
 * @brief The Relax builtin global functions.
 */

#include <string.h>

#include <device/device_api.h>
#include <module/module.h>
#include <relax_vm/relax_vm.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

// Define the relax vm builtin function name prefix.
#define RELAX_VM_BUILTIN_FUNC_NAME(_name_suffix) TVM_RT_WASM_RelaxVM_Builtin_##_name_suffix

// Define the relax vm builtin function prototype.
#define RELAX_VM_FUNC(_name_suffix)                                                                \
    static int RELAX_VM_BUILTIN_FUNC_NAME(_name_suffix)(                                           \
        TVMValue * args_value, const int *args_typecode, int num_args, TVMValue *ret_value,        \
        const int *ret_typecode, void *source_handle)

RELAX_VM_FUNC(AllocaShape) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(MatchShape) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(MakeShape) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckTensorInfo) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckShapeInfo) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckTupleInfo) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(CheckFuncInfo) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(AllocStorage) {
    (void)args_typecode;
    (void)num_args;
    (void)source_handle;

    TVM_RT_WASM_RelaxVirtualMachine vm = args_value[0].v_handle;
    RelaxVMRegisterObject *shape_tuple = args_value[1].v_handle;
    DLDataType dtype = args_value[3].v_type;
    DLDevice dev;
    void *data;

    // device index
    if (args_value[2].v_int64 == -1) {
        dev.device_type = kDLCPU;
    } else {
        dev = vm->devices[0];
    }
    size_t nbytes = TVM_RT_WASM_DLTensor_GetDataBytes(shape_tuple->shape_tuple.shape,
                                                      shape_tuple->shape_tuple.ndim, dtype);
    if (dev.device_type == kDLCPU) {
        data = TVM_RT_WASM_HeapMemoryAlignedAlloc(nbytes);
    } else {
        DeviceAPI *device_api;
        int status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &device_api);
        if (unlikely(status)) {
            return status;
        }
        data = device_api->AllocDataSpace(dev.device_id, nbytes);
    }
    if (unlikely(data == NULL)) {
        return -1;
    }
    RelaxVMRegisterObject *storage_obj;
    TVM_RT_WASM_RelaxVMRegisterCreateObject(storage_obj);

    storage_obj->storage.device = dev;
    storage_obj->storage.data = data;
    ret_value->v_handle = storage_obj;
    *(RelaxVMRegisterTypeCode *)ret_typecode = RelaxVMRegType_VMObjectStorage;
    return 0;
}

RELAX_VM_FUNC(AllocDLTensor) {
    (void)args_typecode;
    (void)num_args;
    (void)source_handle;

    RelaxVMRegisterObject *storage_obj = args_value[0].v_handle;
    int64_t offset = args_value[1].v_int64;
    RelaxVMRegisterObject *shape_obj = args_value[2].v_handle;
    DLDataType dtype = args_value[3].v_type;

    RelaxVMRegisterManagedDLTensor *dl_tensor;
    // set shape_obj and storage_obj
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(dl_tensor);
    dl_tensor->dl_tensor.data = storage_obj->storage.data;
    dl_tensor->dl_tensor.device = storage_obj->storage.device;
    dl_tensor->dl_tensor.ndim = shape_obj->shape_tuple.ndim;
    dl_tensor->dl_tensor.shape = shape_obj->shape_tuple.shape;
    dl_tensor->dl_tensor.dtype = dtype;
    dl_tensor->dl_tensor.strides = NULL;
    dl_tensor->dl_tensor.byte_offset = offset;

    ++storage_obj->ref_num;
    dl_tensor->storage_obj = storage_obj;
    ++shape_obj->ref_num;
    dl_tensor->shape_obj = shape_obj;

    ret_value->v_handle = dl_tensor;
    *(RelaxVMRegisterTypeCode *)ret_typecode = RelaxVMRegType_ManagedDLTensor;
    return 0;
}

RELAX_VM_FUNC(MakeClosure) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(InvokeClosure) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(CallTIRDyn) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    TVM_RT_NOT_IMPLEMENT(-1);
}

RELAX_VM_FUNC(ShapeOf) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(Copy) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(Reshape) {
    (void)num_args;
    (void)source_handle;

    RelaxVMRegisterObject *shape_obj = args_value[1].v_handle;
    RelaxVMRegisterManagedDLTensor *dl_tensor;
    // set shape_obj and storage_obj and should_free_storage (if storage_obj is NULL)
    TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(dl_tensor);

    if (args_typecode[0] == RelaxVMRegType_ManagedDLTensor) {
        RelaxVMRegisterManagedDLTensor *src_tensor = args_value[0].v_handle;
        dl_tensor->dl_tensor = src_tensor->dl_tensor;
        dl_tensor->storage_obj = src_tensor->storage_obj;
        if (dl_tensor->storage_obj) {
            ++dl_tensor->storage_obj->ref_num;
        } else {
            dl_tensor->storage_obj = NULL;
            dl_tensor->should_free_storage = false;
        }
    } else if (args_typecode[0] == RelaxVMRegType_DLTensorHandle) {
        dl_tensor->dl_tensor = *((DLTensor *)(args_value[0].v_handle));
        dl_tensor->storage_obj = NULL;
        dl_tensor->should_free_storage = false;
    }

    dl_tensor->dl_tensor.shape = shape_obj->shape_tuple.shape;
    dl_tensor->dl_tensor.ndim = shape_obj->shape_tuple.ndim;
    ++shape_obj->ref_num;
    dl_tensor->shape_obj = shape_obj;

    ret_value->v_handle = dl_tensor;
    *(RelaxVMRegisterTypeCode *)ret_typecode = RelaxVMRegType_ManagedDLTensor;
    return 0;
}

RELAX_VM_FUNC(ReadIfCond) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(TupleGetItem) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(MakeTuple) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

RELAX_VM_FUNC(TensorToShape) {
    (void)args_value;
    (void)args_typecode;
    (void)num_args;
    (void)ret_value;
    (void)ret_typecode;
    (void)source_handle;
    return 0;
}

int TVM_RT_WASM_RelaxVMRegisterBuiltinGlobalFunctions() {
    static PackedFunction pf[19];
    static int has_registered = 0;

    if (likely(has_registered)) {
        return 0;
    }

    PackedFunction *current_pf = pf;
    int status;

#define REG_FUNC(_func_name_literal, _func_symbol_name_suffix)                                     \
    do {                                                                                           \
        current_pf->exec =                                                                         \
            (TVMBackendPackedCFunc)RELAX_VM_BUILTIN_FUNC_NAME(_func_symbol_name_suffix);           \
        status = TVMFuncRegisterGlobal("vm.builtin."_func_name_literal, (current_pf++), 1);        \
        if (unlikely(status)) {                                                                    \
            TVM_RT_SET_ERROR_RETURN(status, "Cannot register global function vm.builtin.`%s`.",    \
                                    _func_name_literal);                                           \
        }                                                                                          \
    } while (0)

    REG_FUNC("alloc_shape_heap", AllocaShape);
    REG_FUNC("match_shape", MatchShape);
    REG_FUNC("make_shape", MakeShape);

    REG_FUNC("check_tensor_info", CheckTensorInfo);
    REG_FUNC("check_shape_info", CheckShapeInfo);
    REG_FUNC("check_tuple_info", CheckTupleInfo);
    REG_FUNC("check_func_info", CheckFuncInfo);

    REG_FUNC("alloc_storage", AllocStorage);
    REG_FUNC("alloc_tensor", AllocDLTensor);

    REG_FUNC("make_closure", MakeClosure);
    REG_FUNC("invoke_closure", InvokeClosure);
    REG_FUNC("call_tir_dyn", CallTIRDyn);

    REG_FUNC("shape_of", ShapeOf);
    REG_FUNC("copy", Copy);
    REG_FUNC("reshape", Reshape);

    REG_FUNC("read_if_cond", ReadIfCond);
    REG_FUNC("tuple_getitem", TupleGetItem);
    REG_FUNC("make_tuple", MakeTuple);
    REG_FUNC("tensor_to_shape", TensorToShape);

    has_registered = 1;
    return 0;
}
