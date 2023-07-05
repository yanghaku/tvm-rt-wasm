/**
 * @file c_backend_api.c
 * @brief The implementation for tvm/runtime/c_backend_api.h.
 */

#include <tvm/runtime/c_backend_api.h>

#include <device/device_api.h>
#include <module/module.h>

int TVMBackendGetFuncFromEnv(void *mod_node, const char *func_name, TVMFunctionHandle *out) {
    int status =
        TVM_RT_WASM_TrieQuery(((Module *)mod_node)->env_funcs_map, (const uint8_t *)func_name, out);
    if (unlikely(status == TRIE_NOT_FOUND)) {
        status = TVMFuncGetGlobal(func_name, out);
        if (likely(status == TRIE_SUCCESS)) {
            TVM_RT_WASM_TrieInsert(((Module *)mod_node)->env_funcs_map, (const uint8_t *)func_name,
                                   *out);
        }
        if (unlikely(status == TRIE_NOT_FOUND)) {
            TVM_RT_SET_ERROR_RETURN(status, "Cannot find function '%s' from env", func_name);
        }
    }
    return status;
}

void *TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                               int dtype_bits_hint) {
    if (device_type == kDLCPU || device_type == kDLCUDAHost) {
        return TVM_RT_WASM_WorkplaceMemoryAlloc(nbytes);
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return NULL;
    }
    DLDataType type = {dtype_code_hint, dtype_bits_hint, 1};
    return deviceApi->AllocWorkspace(device_id, nbytes, type);
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void *ptr) {
    if (device_type == kDLCPU || device_type == kDLCUDAHost) {
        TVM_RT_WASM_WorkplaceMemoryFree(ptr);
        return 0;
    }
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return -1;
    }
    deviceApi->FreeWorkspace(device_id, ptr);
    return status;
}

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task) {
    (void)num_task;
    // Now WebAssembly does not support threads.
    static TVMParallelGroupEnv parallelGroupEnv = {.num_task = 1, .sync_handle = NULL};
    return flambda(0, &parallelGroupEnv, cdata);
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv *penv) {
    (void)task_id;
    (void)penv;
    return 0;
}
