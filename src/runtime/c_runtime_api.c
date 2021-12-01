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
#include <tvm/runtime/utils/common.h>

/**
 * in this implement:
 *  TVMModuleHandle = Module*
 *  TVMFunctionHandle = TVMBackendPackedCFunc
 */

/*! \brief the global buffer storage */
char global_buf[GLOBAL_BUF_SIZE];

/*! \brief the global function storage, <string,TVMBackendPackedCFunc> */
static Trie *global_functions = NULL;

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
    return ModuleFactory(format, file_name, 0, (Module **)&out);
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
    return ((Module *)mod)->Import((Module *)mod, (Module *)dep);
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
    return ((Module *)mod)->GetFunction((Module *)mod, func_name, query_imports, out);
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
TVM_DLL int TVMModFree(TVMModuleHandle mod) { return ((Module *)mod)->Release((Module *)mod); }

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, nonzero when failure happens
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
    TVMBackendPackedCFunc exec = (TVMBackendPackedCFunc)func;
    // todo: resources for func
    return exec(arg_values, type_codes, num_args, ret_val, ret_type_code, NULL);
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
        return TrieInsert(global_functions, (const uint8_t *)name, f);
    } else {
        void *res;
        int status = TrieQuery(global_functions, (const uint8_t *)name, &res);
        if (unlikely(status == TRIE_NOT_FOUND)) {
            return TrieInsert(global_functions, (const uint8_t *)name, f);
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
    return TrieQuery(global_functions, (const uint8_t *)name, out);
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
TVM_DLL int TVMFuncRemoveGlobal(const char *name) { return TrieInsert(global_functions, (const uint8_t *)name, NULL); }

/*!
 * \brief Create a new runtime stream.
 * \param device_type The device type.
 * \param device_id The device id.
 * \param out The new stream handle.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMStreamCreate(int device_type, int device_id, TVMStreamHandle *out) {
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    *out = deviceApi->CreateStream(deviceApi, device_id);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->FreeStream(deviceApi, device_id, stream);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->SetStream(deviceApi, device_id, handle);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->StreamSync(deviceApi, device_id, stream);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->SyncStreamFromTo(deviceApi, device_id, src, dst);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    *out_data = deviceApi->AllocDataSpace(deviceApi, dev.device_id, nbytes, alignment, type_hint);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    *out_data = deviceApi->AllocDataSpaceScope(deviceApi, dev.device_id, ndim, shape, dtype, mem_scope);
    return status;
}

/*!
 * \brief Free a data space on device.
 * \param dev The device to perform operation.
 * \param ptr The data space.
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDeviceFreeDataSpace(DLDevice dev, void *ptr) {
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(dev.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->FreeDataSpace(deviceApi, dev.device_id, ptr);
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
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(from->device.device_type, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->CopyDataFromTo(deviceApi, from, to, stream);
    return status;
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
