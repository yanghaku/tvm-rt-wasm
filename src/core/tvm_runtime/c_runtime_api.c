/**
 * @file c_runtime_api.c
 * @brief The implementation for tvm/runtime/c_runtime_api.h.
 */

#include <string.h>
#include <tvm/runtime/c_runtime_api.h>

#include <device/device_api.h>
#include <module/module.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

/**
 * In this implementation:
 *  TVMModuleHandle = Module*
 *
 *  TVMFunctionHandle = PackedFunction*
 *
 *  @sa module/module.h
 */

/*--------------------------------------global variables------------------------------------------*/

/** @brief the global buffer storage */
char global_buf[GLOBAL_BUF_SIZE];

/** @brief the global function map: <string, PackedFunction*>. */
static Trie *global_functions = NULL;

/*---------------------------------------public functions-----------------------------------------*/

void TVMAPISetLastError(const char *msg) { strcpy(global_buf, msg); }

const char *TVMGetLastError(void) { return global_buf; }

int TVMModLoadFromFile(const char *file_name, const char *format, TVMModuleHandle *out) {
    CHECK_INPUT_POINTER(file_name, -2, "Filename");
    CHECK_INPUT_POINTER(format, -2, "Format");
    CHECK_INPUT_POINTER(out, -2, "TVMModuleHandle pointer");
    if (strcmp(format, "so") == 0 || strcmp(format, "dll") == 0 || strcmp(format, "dylib") == 0 ||
        strcmp(format, "dso") == 0) {
        return TVM_RT_WASM_SharedLibraryModuleCreate(file_name, (Module **)out);
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported module file format `%s`\n", format);
}

int TVMModImport(TVMModuleHandle mod, TVMModuleHandle dep) {
    (void)mod;
    (void)dep;
    TVM_RT_NOT_IMPLEMENT(-1);
}

int TVMModGetFunction(TVMModuleHandle mod, const char *func_name, int query_imports,
                      TVMFunctionHandle *out) {
    CHECK_INPUT_POINTER(mod, -2, "TVMModuleHandle");
    CHECK_INPUT_POINTER(func_name, -2, "Function Name");
    CHECK_INPUT_POINTER(out, -2, "Out FunctionHandle pointer");

    Module *m = (Module *)mod;
    return m->GetFunction(m, func_name, query_imports, (PackedFunction **)out);
}

int TVMModFree(TVMModuleHandle mod) {
    CHECK_INPUT_POINTER(mod, -2, "TVMModuleHandle");
    // todo: Prevent being free when it is still being used
    return ((Module *)mod)->Release((Module *)mod);
}

/**
 * @brief In this implementation, TVMFunctionHandle do not need to be freed.
 */
int TVMFuncFree(TVMFunctionHandle func) {
    (void)func;
    return 0;
}

int TVMFuncCall(TVMFunctionHandle func, TVMValue *arg_values, int *type_codes, int num_args,
                TVMValue *ret_val, int *ret_type_code) {
    CHECK_INPUT_POINTER(func, -2, "TVMFunctionHandle");
    CHECK_INPUT_POINTER(arg_values, -2, "TVM argument values");
    CHECK_INPUT_POINTER(type_codes, -2, "TVM argument types");
    CHECK_INPUT_POINTER(ret_val, -2, "TVM return values");
    CHECK_INPUT_POINTER(ret_type_code, -2, "TVM return types");

    PackedFunction *pf = (PackedFunction *)func;
    return pf->exec(arg_values, type_codes, num_args, ret_val, ret_type_code, func);
}

int TVMFuncRegisterGlobal(const char *name, TVMFunctionHandle f, int override) {
    CHECK_INPUT_POINTER(name, -2, "Function name");
    CHECK_INPUT_POINTER(f, -2, "TVMFunctionHandle");
    if (override) {
        return TVM_RT_WASM_TrieInsert(global_functions, (const uint8_t *)name, f);
    } else {
        void *res;
        int status = TVM_RT_WASM_TrieQuery(global_functions, (const uint8_t *)name, &res);
        if (unlikely(status == TRIE_NOT_FOUND)) {
            return TVM_RT_WASM_TrieInsert(global_functions, (const uint8_t *)name, f);
        }
        return status;
    }
}

int TVMFuncGetGlobal(const char *name, TVMFunctionHandle *out) {
    CHECK_INPUT_POINTER(name, -2, "Function name");
    CHECK_INPUT_POINTER(out, -2, "TVMFunctionHandle pointer");
    return TVM_RT_WASM_TrieQuery(global_functions, (const uint8_t *)name, out);
}

int TVMFuncListGlobalNames(int *out_size, const char ***out_array) {
    CHECK_INPUT_POINTER(out_size, -2, "The number of functions");
    CHECK_INPUT_POINTER(out_array, -2, "Function array");
    *out_size = 0;
    TVM_RT_NOT_IMPLEMENT(-1);
}

int TVMFuncRemoveGlobal(const char *name) {
    CHECK_INPUT_POINTER(name, -2, "Function name");
    TVM_RT_NOT_IMPLEMENT(-1);
}

int TVMArrayAlloc(const tvm_index_t *shape, int ndim, int dtype_code, int dtype_bits,
                  int dtype_lanes, int device_type, int device_id, TVMArrayHandle *out) {
    CHECK_INPUT_POINTER(shape, -2, "Shape");
    CHECK_INPUT_POINTER(out, -2, "TVMArrayHandle pointer");

    DLDevice d = {
        .device_id = device_id,
        .device_type = device_type,
    };
    DLDataType tp = {
        .code = dtype_code,
        .bits = dtype_bits,
        .lanes = dtype_lanes,
    };
    size_t bytes = 1;
    for (int i = 0; i < ndim; ++i) {
        bytes *= shape[i];
    }
    void *data = NULL;
    int status = TVMDeviceAllocDataSpace(d, bytes, 0, tp, &data);
    if (unlikely(status)) {
        return status;
    }

    DLTensor *t = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLTensor));
    t->device = d;
    t->dtype = tp;
    t->data = data;
    t->byte_offset = 0;
    t->strides = NULL;
    t->ndim = ndim;
    t->shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * ndim);
    for (int i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }

    *out = t;
    return 0;
}

int TVMArrayFree(TVMArrayHandle handle) {
    CHECK_INPUT_POINTER(handle, -2, "TVMArrayHandle");
    int status = TVMDeviceFreeDataSpace(handle->device, handle->data);
    TVM_RT_WASM_HeapMemoryFree(handle->shape);
    TVM_RT_WASM_HeapMemoryFree(handle);
    return status;
}

int TVMArrayCopyFromBytes(TVMArrayHandle handle, void *data, size_t nbytes) {
    CHECK_INPUT_POINTER(handle, -2, "TVMArrayHandle");
    CHECK_INPUT_POINTER(data, -2, "CPU Data pointer");

    if (handle->device.device_type == kDLCPU) {
        uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(handle);
        memcpy(handle->data, data, MIN(bytes, nbytes));
        return 0;
    } else {
        DeviceAPI *deviceApi;
        int status = TVM_RT_WASM_DeviceAPIGet(handle->device.device_type, &deviceApi);
        if (unlikely(status)) {
            return status;
        }
        int64_t shape[] = {(int64_t)nbytes};
        DLTensor from = {
            .device = {.device_type = kDLCPU},
            .dtype = handle->dtype,
            .shape = shape,
            .ndim = 1,
        };
        return deviceApi->CopyDataFromTo(&from, handle, NULL);
    }
}

int TVMArrayCopyToBytes(TVMArrayHandle handle, void *data, size_t nbytes) {
    CHECK_INPUT_POINTER(handle, -2, "TVMArrayHandle");
    CHECK_INPUT_POINTER(data, -2, "CPU Data pointer");

    if (handle->device.device_type == kDLCPU) {
        uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(handle);
        memcpy(data, handle->data, MIN(bytes, nbytes));
        return 0;
    } else {
        DeviceAPI *deviceApi;
        int status = TVM_RT_WASM_DeviceAPIGet(handle->device.device_type, &deviceApi);
        if (unlikely(status)) {
            return status;
        }
        int64_t shape[] = {(int64_t)nbytes};
        DLTensor to = {
            .device = {.device_type = kDLCPU},
            .dtype = handle->dtype,
            .shape = shape,
            .ndim = 1,
        };
        return deviceApi->CopyDataFromTo(handle, &to, NULL);
    }
}

int TVMArrayCopyFromTo(TVMArrayHandle from, TVMArrayHandle to, TVMStreamHandle stream) {
    return TVMDeviceCopyDataFromTo(from, to, stream);
}

int TVMArrayFromDLPack(DLManagedTensor *from, TVMArrayHandle *out) {
    CHECK_INPUT_POINTER(from, -2, "From DLManagedTensor");
    CHECK_INPUT_POINTER(out, -2, "TVMArrayHandle pointer");
    TVM_RT_NOT_IMPLEMENT(-1);
}

int TVMArrayToDLPack(TVMArrayHandle from, DLManagedTensor **out) {
    CHECK_INPUT_POINTER(from, -2, "From TVMArrayHandle");
    CHECK_INPUT_POINTER(out, -2, "DLManagedTensor pointer");
    TVM_RT_NOT_IMPLEMENT(-1);
}

void TVMDLManagedTensorCallDeleter(DLManagedTensor *dltensor) {
    if (dltensor) {
        (*(dltensor->deleter))(dltensor);
    }
}

int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle *out) {
    CHECK_INPUT_POINTER(out, -2, "TVMStreamHandle pointer");
    if (device_type == kDLCPU) {
        *out = NULL;
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    *out = deviceApi->CreateStream(device_id);
    return *out == NULL; // if successful, *out is not null, return 0
}

int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->FreeStream(device_id, stream);
}

int TVMSetStream(int device_type, int device_id, TVMStreamHandle handle) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->SetStream(device_id, handle);
}

int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->StreamSync(device_id, stream);
}

int TVMStreamStreamSynchronize(int device_type, int device_id, TVMStreamHandle src,
                               TVMStreamHandle dst) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->SyncStreamFromTo(device_id, src, dst);
}

int TVMDeviceAllocDataSpace(DLDevice dev, size_t nbytes, size_t alignment, DLDataType type_hint,
                            void **out_data) {
    CHECK_INPUT_POINTER(out_data, -2, "Output data pointer");
    if (dev.device_type == kDLCPU || dev.device_type == kDLCUDAHost) {
        *out_data = TVM_RT_WASM_HeapMemoryAlloc(nbytes);
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    *out_data = deviceApi->AllocDataSpace(dev.device_id, nbytes, alignment, type_hint);
    return *out_data == NULL;
}

int TVMDeviceAllocDataSpaceWithScope(DLDevice dev, int ndim, const int64_t *shape, DLDataType dtype,
                                     const char *mem_scope, void **out_data) {
    (void)dev;
    (void)ndim;
    (void)shape;
    (void)dtype;
    CHECK_INPUT_POINTER(mem_scope, -2, "Memory scope");
    CHECK_INPUT_POINTER(out_data, -2, "Out data pointer");
    TVM_RT_NOT_IMPLEMENT(-1);
}

int TVMDeviceFreeDataSpace(DLDevice dev, void *ptr) {
    if (dev.device_type == kDLCPU || dev.device_type == kDLCUDAHost) {
        TVM_RT_WASM_HeapMemoryFree(ptr);
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->FreeDataSpace(dev.device_id, ptr);
}

int TVMDeviceCopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
    CHECK_INPUT_POINTER(from, -2, "From DLTensor");
    CHECK_INPUT_POINTER(to, -2, "To DLTensor");
    DLDeviceType type;
    DeviceAPI *deviceApi;

    if (from->device.device_type == to->device.device_type || from->device.device_type == kDLCPU) {
        // same device or from is cpu
        type = to->device.device_type;
    } else if (to->device.device_type == kDLCPU) { // to is cpu
        type = from->device.device_type;
    } else if ((from->device.device_type == kDLCUDA && to->device.device_type == kDLCUDAHost) ||
               (from->device.device_type == kDLCUDAHost && to->device.device_type == kDLCUDA)) {
        type = kDLCUDA;
    } else {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "Unsupported data copy: from device_type(%d) to device_type(%d)",
                                from->device.device_type, to->device.device_type);
    }

    if (type == kDLCPU) {
        uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(from);
        memcpy(to->data, from->data, bytes);
        return 0;
    }

    int status = TVM_RT_WASM_DeviceAPIGet(type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    return deviceApi->CopyDataFromTo(from, to, stream);
}

int TVM_RT_WASM_SetDevice(TVMValue *args, const int *_tc, int _n, TVMValue *_rv, const int *_rt,
                          void *_h) {
    (void)_tc;
    (void)_n;
    (void)_rv;
    (void)_rt;
    (void)_h;
    if (args->v_device.device_type == kDLCPU) {
        return 0;
    }
    static DLDevice cur_device = {.device_type = kDLCPU, .device_id = 0};
    if (cur_device.device_type == args->v_device.device_type &&
        cur_device.device_id == args->v_device.device_id) {
        return 0;
    }

    DeviceAPI *api = NULL;
    TVM_RT_WASM_DeviceAPIGet(args->v_device.device_type, &api);
    cur_device = args->v_device;
    return api->SetDevice(args->v_device.device_id);
}

static TVM_ATTRIBUTE_UNUSED __attribute__((constructor)) void TVM_RT_WASM_Constructor() {
    static PackedFunction pf[] = {
        {.exec = (TVMBackendPackedCFunc)TVM_RT_WASM_SetDevice, .module = NULL, .reserved = 0}};

    TVM_RT_WASM_TrieCreate(&global_functions);
    if (unlikely(TVM_RT_WASM_TrieInsert(global_functions, (const uint8_t *)TVM_SET_DEVICE_FUNCTION,
                                        pf)) != 0) {
        fprintf(stderr, "Register global function fail!\n");
        exit(-1);
    }
}

static TVM_ATTRIBUTE_UNUSED __attribute__((destructor)) void TVM_RT_WASM_Destructor() {
    // release global functions
    if (global_functions) {
        TVM_RT_WASM_TrieRelease(global_functions);
    }

    // release the devices instance
    TVM_RT_WASM_DeviceReleaseAll();
}

/*----------------The following API will not be implemented in this project-----------------------*/

/*
int TVMCFuncSetReturn(TVMRetValueHandle ret, TVMValue *value, int *type_code, int num_ret);
int TVMCbArgToReturn(TVMValue *value, int *code);
int TVMFuncCreateFromCFunc(TVMPackedCFunc func, void *resource_handle, TVMPackedCFuncFinalizer fin,
                           TVMFunctionHandle *out);
void TVMDLManagedTensorCallDeleter(DLManagedTensor *dltensor) {}
int TVMObjectGetTypeIndex(TVMObjectHandle obj, unsigned *out_type_index);
int TVMObjectTypeKey2Index(const char *type_key, unsigned *out_type_index);
int TVMObjectTypeIndex2Key(unsigned tindex, char **out_type_key);
int TVMObjectRetain(TVMObjectHandle obj);
int TVMObjectFree(TVMObjectHandle obj);
int TVMByteArrayFree(TVMByteArray *arr);
int TVMObjectDerivedFrom(uint32_t child_type_index, uint32_t parent_type_index, int *is_derived);

// in c_backend_api.h
int TVMBackendRegisterEnvCAPI(const char *name, void *ptr);
int TVMBackendRunOnce(void **handle, int (*f)(void *), void *cdata, int nbytes);
*/
