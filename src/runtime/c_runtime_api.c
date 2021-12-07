/*!
 * \file src/runtime/c_runtime_api.c
 * \brief the implement for c_runtime_api.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device/device_api.h>
#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/cuda_common.h>
#include <tvm/runtime/utils/tensor_helper.h>

/**
 * in this implement:
 *  TVMModuleHandle = Module*
 *
 *  TVMFunctionHandle = PackedFunction*
 *
 *  \sa module.h
 */

/**-------------------------------------global variables--------------------------------------------------------------*/

/*! \brief the global buffer storage */
char global_buf[GLOBAL_BUF_SIZE];

/*! \brief the global function storage, <string,PackedFunction*> */
static Trie *global_functions = NULL;

/**--------------------------------------public functions-------------------------------------------------------------*/

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
TVM_DLL void TVMAPISetLastError(const char *msg) { strcpy(global_buf, msg); }

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and nonzero when an error occurred,
 *  TVMGetLastError can be called to retrieve the error
 *
 *  this function is thread safe and can be called by different thread
 *  \return error info
 */
TVM_DLL const char *TVMGetLastError(void) { return global_buf; }

/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, nonzero when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by TVMModImport.
 */
TVM_DLL int TVMModLoadFromFile(const char *file_name, const char *format, TVMModuleHandle *out) {
    return TVM_RT_WASM_ModuleFactory(format, file_name, MODULE_FACTORY_RESOURCE_FILE, (Module **)&out);
}

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMModImport(TVMModuleHandle mod, TVMModuleHandle dep) {
    // todo: implement it
    return -1;
}

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, nonzero when failure happens
 */
TVM_DLL int TVMModGetFunction(TVMModuleHandle mod, const char *func_name, int query_imports, TVMFunctionHandle *out) {
    Module *m = (Module *)mod;
    int status = TVM_RT_WASM_TrieQuery(m->module_funcs_map, (const uint8_t *)func_name, out);
    if (likely(status != TRIE_NOT_FOUND)) {
        return status;
    }

    if (query_imports) {
        status = TVM_RT_WASM_TrieQuery(m->env_funcs_map, (const uint8_t *)func_name, out);
    }
    return status;
}

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active TVMFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until TVMFuncFree is called.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMModFree(TVMModuleHandle mod) {
    // todo: Prevent being free when it is still being used
    return ((Module *)mod)->Release((Module *)mod);
}

/*!
 * \brief in this implement, TVMFunctionHandle do not need to be free
 * \param func The function handle
 * \return 0
 */
TVM_DLL int TVMFuncFree(TVMFunctionHandle func) { return 0; }

/*!
 * \brief Call a Packed TVM Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, nonzero when failure happens
 * \note TVM calls always exchanges with type bits=64, lanes=1
 */
TVM_DLL int TVMFuncCall(TVMFunctionHandle func, TVMValue *arg_values, int *type_codes, int num_args, TVMValue *ret_val,
                        int *ret_type_code) {
    PackedFunction *pf = (PackedFunction *)func;
    return pf->exec(arg_values, type_codes, num_args, ret_val, ret_type_code, func);
}

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
TVM_DLL int TVMFuncRegisterGlobal(const char *name, TVMFunctionHandle f, int override) {
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

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by TVM runtime,
 *  So TVMFuncFree is should not be called when it get deleted.
 */
TVM_DLL int TVMFuncGetGlobal(const char *name, TVMFunctionHandle *out) {
    return TVM_RT_WASM_TrieQuery(global_functions, (const uint8_t *)name, out);
}

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMFuncListGlobalNames(int *out_size, const char ***out_array) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Remove a global function.
 * \note The implement here is to replace it with NULL
 * \param name The name of the function.
 */
TVM_DLL int TVMFuncRemoveGlobal(const char *name) {
    return TVM_RT_WASM_TrieInsert(global_functions, (const uint8_t *)name, NULL);
}

/*!
 * \brief Create a new runtime stream.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param out The new stream handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle *out) {
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
    return status;
}

/*!
 * \brief Free a created stream handle.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param stream The stream to be freed.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamFree(int device_type, int device_id, TVMStreamHandle stream) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->FreeStream(device_id, stream);
    return status;
}

/*!
 * \brief Set the runtime stream of current thread to be stream.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param handle The stream handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMSetStream(int device_type, int device_id, TVMStreamHandle handle) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->SetStream(device_id, handle);
    return status;
}

/*!
 * \brief Wait until all computations on stream completes.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param stream The stream to be synchronized.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMSynchronize(int device_type, int device_id, TVMStreamHandle stream) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->StreamSync(device_id, stream);
    return status;
}

/*!
 * \brief Synchronize two streams of execution.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamStreamSynchronize(int device_type, int device_id, TVMStreamHandle src, TVMStreamHandle dst) {
    if (device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->SyncStreamFromTo(device_id, src, dst);
    return status;
}

/*!
 * \brief Allocate a data space on device.
 * \param dev The device to perform operation.
 * \param nbytes The number of bytes in memory.
 * \param alignment The alignment of the memory.
 * \param type_hint The type of elements.
 * \param out_data The allocated device pointer.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceAllocDataSpace(DLDevice dev, size_t nbytes, size_t alignment, DLDataType type_hint,
                                    void **out_data) {
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
    return status;
}

/*!
 * \brief Allocate a data space on device with special memory scope.
 * \param dev The device to perform operation.
 * \param ndim The number of dimension of the tensor.
 * \param shape The shape of the tensor.
 * \param dtype The type of elements.
 * \param mem_scope The memory scope of the tensor can be nullptr, which indicate the default global DRAM
 * \param out_data The allocated device pointer.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceAllocDataSpaceWithScope(DLDevice dev, int ndim, const int64_t *shape, DLDataType dtype,
                                             const char *mem_scope, void **out_data) {
    SET_ERROR_RETURN(-1, "unsupported yet");
}

/*!
 * \brief Free a data space on device.
 * \param dev The device to perform operation.
 * \param ptr The data space.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceFreeDataSpace(DLDevice dev, void *ptr) {
    if (dev.device_type == kDLCPU || dev.device_type == kDLCUDAHost) {
        TVM_RT_WASM_HeapMemoryFree(ptr);
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->FreeDataSpace(dev.device_id, ptr);
    return status;
}

/*!
 * \brief Copy data from one place to another.
 * \param from The source tensor.
 * \param to The target tensor.
 * \param stream Optional stream object.
 * \return 0 when success, nonzero when failure happens.
 */
TVM_DLL int TVMDeviceCopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
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
        SET_ERROR_RETURN(-1, "unsupported operator: copy data from device_type(%d) to device_typ(%d)\n",
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
    deviceApi->CopyDataFromTo(from, to, stream);
    return status;
}

int TVM_RT_WASM_SetDevice(TVMValue *args, int *_tc, int _n, TVMValue *_rv, int *_rt, void *_h) {
    if (args->v_device.device_type == kDLCPU) {
        return 0;
    }
    DeviceAPI *api;
    TVM_RT_WASM_DeviceAPIGet(args->v_device.device_type, &api);
    api->SetDevice(args->v_device.device_id);
    return 0;
}

static __attribute__((constructor)) void tvm_runtime_for_webassembly_constructor() {
    static PackedFunction pf[1] = {{.exec = TVM_RT_WASM_SetDevice, .module = NULL, .reserved = 0}};

    TVM_RT_WASM_TrieCreate(&global_functions);
    if (unlikely(TVM_RT_WASM_TrieInsert(global_functions, (const uint8_t *)TVM_SET_DEVICE_FUNCTION, pf)) != 0) {
        fprintf(stderr, "register global function fail!\n");
        exit(-1);
    }
}

static __attribute__((destructor)) void tvm_runtime_for_webassembly_destructor() {
    // release global functions
    if (global_functions) {
        TVM_RT_WASM_TrieRelease(global_functions);
    }

    // if sys_lib_modules, release it
    Module *sys_lib;
    if (TVM_RT_WASM_ModuleFactory(MODULE_SYSTEM_LIB, 0, 0, &sys_lib) == 0) {
        sys_lib->Release(sys_lib);
    }

    // release the devices instance
    TVM_RT_WASM_DeviceReleaseAll();
}

/**-------------------------The following API will not be implemented in this project---------------------------------*/

/*
int TVMCFuncSetReturn(TVMRetValueHandle ret, TVMValue *value, int *type_code, int num_ret);
int TVMCbArgToReturn(TVMValue *value, int *code);
int TVMFuncCreateFromCFunc(TVMPackedCFunc func, void *resource_handle, TVMPackedCFuncFinalizer fin,
                           TVMFunctionHandle *out);
int TVMArrayAlloc(const tvm_index_t *shape, int ndim, int dtype_code, int dtype_bits, int dtype_lanes, int device_type,
                  int device_id, TVMArrayHandle *out);
int TVMArrayFree(TVMArrayHandle handle);
int TVMArrayCopyFromBytes(TVMArrayHandle handle, void *data, size_t nbytes);
int TVMArrayCopyToBytes(TVMArrayHandle handle, void *data, size_t nbytes);
int TVMArrayCopyFromTo(TVMArrayHandle from, TVMArrayHandle to, TVMStreamHandle stream);
int TVMArrayFromDLPack(DLManagedTensor *from, TVMArrayHandle *out);
int TVMArrayToDLPack(TVMArrayHandle from, DLManagedTensor **out);
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
