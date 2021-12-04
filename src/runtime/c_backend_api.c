/*!
 * \file src/runtime/c_backend_api.c
 * \brief the implement for c_backend_api.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/device/device_api.h>
#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/common.h>

/*!
 * \brief Backend function for modules to get function
 *  from its environment mod_node (its imports and global function).
 *  The user do should not call TVMFuncFree on func.
 *
 * \param mod_node The module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendGetFuncFromEnv(void *mod_node, const char *func_name, TVMFunctionHandle *out) {
    int status = TrieQuery(((Module *)mod_node)->env_funcs_map, (const uint8_t *)func_name, out);
    if (unlikely(status == TRIE_NOT_FOUND)) {
        status = TVMFuncGetGlobal(func_name, out);
        if (likely(status == TRIE_SUCCESS)) {
            TrieInsert(((Module *)mod_node)->env_funcs_map, (const uint8_t *)func_name, *out);
        }
    }
    return status;
}

/*!
 * \brief Backend function to register system-wide library symbol.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
    if (unlikely(system_lib_symbol == NULL)) {
        int status = TrieCreate(&system_lib_symbol);
        if (unlikely(status)) {
            fprintf(stderr, "%s:create a new trie node Error", __FUNCTION__);
            return -1;
        }
    }
    return TrieInsert(system_lib_symbol, (const uint8_t *)name, ptr);
}

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocated space is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
TVM_DLL void *TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                                       int dtype_bits_hint) {
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return NULL;
    }
    DLDataType type = {dtype_code_hint, dtype_bits_hint, 1};
    return deviceApi->AllocWorkspace(device_id, nbytes, type);
}

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
TVM_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void *ptr) {
    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(device_type, &deviceApi);
    if (unlikely(status)) {
        return -1;
    }
    deviceApi->FreeWorkspace(device_id, ptr);
    return status;
}

/*!
 * \brief Backend function for running parallel jobs.
 *
 * \param flambda The parallel function to be launched.
 * \param cdata The closure data.
 * \param num_task Number of tasks to launch, can be 0, means launch with all available threads.
 *
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task) {
    // Now WebAssembly does not support threads.
    static TVMParallelGroupEnv parallelGroupEnv = {.num_task = 1, .sync_handle = NULL};
    return flambda(0, &parallelGroupEnv, cdata);
}

/*!
 * \brief BSP barrrier between parallel threads
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv *penv) { return 0; }
