/**
 * @file cuda/cuda_module.c
 * @brief Implement functions for cuda module.
 */

#include <cuda_common.h>
#include <device/cpu_memory.h>
#include <device/device_api.h>
#include <module/function_info.h>
#include <module/module_impl.h>
#include <utils/binary_reader.h>

/** @brief CUDA function information, derive from PackedFunction. */
typedef struct CUDAFunctionInfo {
    /** @brief The function pointer to execute. */
    TVMBackendPackedCFunc exec;

    /** @brief base information */
    BASE_FUNCTION_INFO

    /** @brief the cuda functions in cuda module */
    CUfunction cu_function;
} CUDAFunctionInfo;

/** @brief define the cuda module derived from module */
typedef struct CUDAModule {
    MODULE_BASE_MEMBER

    /** @brief PackedFunctions in this module, with its information. */
    CUDAFunctionInfo *functions;
    size_t num_functions;

    // todo: change it to support multi-GPU
    /** @brief the cuda module */
    CUmodule cu_module;
} CUDAModule;

static int TVM_RT_WASM_CUDAWrappedFunction(TVMValue *args, const int *type_codes, int num_args,
                                           TVMValue *ret_val, const int *ret_type_codes,
                                           void *resource_handle) {
    (void)ret_val;
    (void)ret_type_codes;

    CUDAFunctionInfo *info = (CUDAFunctionInfo *)resource_handle;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;

    uint32_t num_kernel_args = info->num_kernel_args;
    CHECK_DYN_MEM();
    CHECK_AND_GET_DIM();
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        info->kernel_arg_storages[i] = &args[i].v_handle;
    }

    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &deviceApi);
    if (unlikely(status)) {
        return status;
    }

    CUstream stream = (CUstream)deviceApi->GetStream();
    CUDA_DRIVER_CALL(cuLaunchKernel(info->cu_function, grid_dim[0], grid_dim[1], grid_dim[2],
                                    block_dim[0], block_dim[1], block_dim[2], dyn_shared_mem_size,
                                    stream, info->kernel_arg_storages, NULL));

    return status;
}

static int TVM_RT_WASM_CUDAModuleReleaseFunc(Module *self) {
    CUDAModule *c = (CUDAModule *)self;
    MODULE_BASE_MEMBER_FREE(c);

    for (size_t i = 0; i < c->num_functions; ++i) {
        if (c->functions[i].func_arg_index_map) {
            TVM_RT_WASM_HeapMemoryFree(c->functions[i].func_arg_index_map);
        }
        if (c->functions[i].kernel_arg_storages) {
            TVM_RT_WASM_HeapMemoryFree(c->functions[i].kernel_arg_storages);
        }
    }
    TVM_RT_WASM_HeapMemoryFree(c->functions);

    cuModuleUnload(c->cu_module);
    // free self
    TVM_RT_WASM_HeapMemoryFree(c);
    return 0;
}

static void TVM_RT_WASM_CUDAModuleAllocate(CUDAModule **cudaModule, size_t num_func) {
    *cudaModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAModule));
    memset(*cudaModule, 0, sizeof(CUDAModule));
    (*cudaModule)->Release = TVM_RT_WASM_CUDAModuleReleaseFunc;
    (*cudaModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*cudaModule)->module_funcs_map));
    (*cudaModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAFunctionInfo) * num_func);
    memset((*cudaModule)->functions, 0, sizeof(CUDAFunctionInfo) * num_func);
    (*cudaModule)->num_functions = num_func;
    for (size_t fid = 0; fid < num_func; ++fid) {
        (*cudaModule)->functions[fid].exec = (TVMBackendPackedCFunc)TVM_RT_WASM_CUDAWrappedFunction;
    }
}

/**
 * @brief Create a CUDA module instance from the byte stream.
 * @param reader The module binary reader.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_CUDAModuleCreate(BinaryReader *reader, Module **out) {
    *out = NULL;
    const char *cur_ptr;
    int status = -1;

    // parse format
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t fmt_size = (size_t) * (uint64_t *)cur_ptr; // format string size
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, fmt_size, fail_label);
    if (!((fmt_size == 5 && memcmp(cur_ptr, "cubin", fmt_size) == 0) ||
          (fmt_size == 3 && memcmp(cur_ptr, "ptx", fmt_size) == 0))) {
        TVM_RT_SET_ERROR_RETURN(-1, "Unsupported binary format %s", cur_ptr);
    }

    // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t func_map_size = (size_t) * (uint64_t *)cur_ptr; // func_map_size
    // allocate memory for this
    TVM_RT_WASM_CUDAModuleAllocate((CUDAModule **)out, func_map_size);
    CUDAModule *cuda_module = *(CUDAModule **)out;

    const char *names = reader->current_ptr; // for init functions from cu_module

    for (size_t fid = 0; fid < func_map_size; ++fid) {
        CUDAFunctionInfo *info = cuda_module->functions + fid;
        PARSE_FUNC_INFO(cuda_module, cur_ptr, fail_label);
    }

    // parse data
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t source_len = (size_t) * (uint64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, source_len, fail_label);

    // init cu_module
    DeviceAPI *cuda_dev_api;
    status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &cuda_dev_api);
    if (unlikely(status)) {
        goto fail_label;
    }
    cuda_dev_api->SetDevice(0);
    CUDA_DRIVER_CALL_OR_GOTO_WITH_STATUS(cuModuleLoadData(&cuda_module->cu_module, cur_ptr),
                                         fail_label);

    // load cu_functions
    for (size_t fid = 0; fid < func_map_size; ++fid) {
        // The name is checked before.
        // key: name
        size_t name_size = (size_t) * (uint64_t *)names;
        names += sizeof(uint64_t); // name_size

        CUDA_DRIVER_CALL_OR_GOTO_WITH_STATUS(
            cuModuleGetFunction(&cuda_module->functions[fid].cu_function, cuda_module->cu_module,
                                names),
            fail_label);

        names += name_size; // name string

        // functionInfo.name
        name_size = (size_t) * (uint64_t *)names;
        names += sizeof(uint64_t) + name_size; // name_size + name string
        names += sizeof(uint64_t);             // num_func_args
        names += cuda_module->functions[fid].num_kernel_args * sizeof(DLDataType); // arg types

        size_t mp_size = (size_t) * (uint64_t *)names;
        names += sizeof(uint64_t); // num_func_arg_map
        for (size_t i = 0; i < mp_size; ++i) {
            name_size = (size_t) * (uint64_t *)names;
            names += sizeof(uint64_t) + name_size; // name_size + name string
        }
    }
    return 0;

fail_label:
    if (*out) {
        TVM_RT_WASM_CUDAModuleReleaseFunc(*out);
        *out = NULL;
    }
    return status;
}
