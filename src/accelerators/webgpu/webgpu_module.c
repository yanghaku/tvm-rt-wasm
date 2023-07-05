/**
 * @file module/webgpu_module.c
 * @brief Implement functions for WebGPU module.
 */

#include <device/device_api.h>
#include <module/function_info.h>
#include <module/module_impl.h>
#include <webgpu_common.h>

typedef struct WebGPUFunctionInfo {
    BASE_FUNCTION_INFO

    WGPU_Function device_func;
} WebGPUFunctionInfo;

/** @brief define the WebGPU module derived from module */
typedef struct WebGPUModule {
    MODULE_BASE_MEMBER

    /** @brief the WebGPU module */
    // todo: multi-GPU support
    WebGPUFunctionInfo *functions;
    size_t num_functions;
} WebGPUModule;

static int TVM_RT_WASM_WebGPUWrappedFunction(TVMValue *args, const int *type_codes, int num_args,
                                             TVMValue *ret_val, const int *ret_type_codes,
                                             void *resource_handle) {
    (void)ret_val;
    (void)ret_type_codes;
    PackedFunction *pf = (PackedFunction *)resource_handle;
    int func_id = (int)pf->reserved;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    WebGPUFunctionInfo *info = ((WebGPUModule *)pf->module)->functions + func_id;

    uint32_t num_kernel_args = info->num_kernel_args;
    CHECK_DYN_MEM();
    if (dyn_shared_mem_size != 0) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "WebGPU cannot support dynamic shared memory, but got size %zu.",
                                dyn_shared_mem_size);
    }

    CHECK_AND_GET_DIM();
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        info->kernel_arg_storages[i] = args[i].v_handle;
    }

    (void)block_dim;
    int status = WGPU_FunctionRun(info->device_func, (WGPU_Memory *)info->kernel_arg_storages,
                                  num_kernel_args, grid_dim[0], grid_dim[1], grid_dim[2]);

    return status;
}

static int TVM_RT_WASM_WebGPUModuleReleaseFunc(Module *self) {
    WebGPUModule *w = (WebGPUModule *)self;
    MODULE_BASE_MEMBER_FREE(w);

    for (size_t i = 0; i < w->num_functions; ++i) {
        if (w->functions[i].func_arg_index_map) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].func_arg_index_map);
        }
        if (w->functions[i].kernel_arg_storages) {
            TVM_RT_WASM_HeapMemoryFree(w->functions[i].kernel_arg_storages);
        }
        if (w->functions[i].device_func) {
            WGPU_FunctionFree(w->functions[i].device_func);
        }
    }
    TVM_RT_WASM_HeapMemoryFree(w->functions);
    TVM_RT_WASM_HeapMemoryFree(w->packed_functions);

    // free self
    TVM_RT_WASM_HeapMemoryFree(w);
    return 0;
}

static void TVM_RT_WASM_WebGPUModuleAllocate(WebGPUModule **webgpuModule, size_t num_func) {
    *webgpuModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUModule));
    memset(*webgpuModule, 0, sizeof(WebGPUModule));
    (*webgpuModule)->Release = TVM_RT_WASM_WebGPUModuleReleaseFunc;
    (*webgpuModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*webgpuModule)->module_funcs_map));
    (*webgpuModule)->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_func);
    (*webgpuModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUFunctionInfo) * num_func);
    memset((*webgpuModule)->functions, 0, sizeof(WebGPUFunctionInfo) * num_func);
    (*webgpuModule)->num_functions = num_func;
    for (size_t fid = 0; fid < num_func; ++fid) {
        (*webgpuModule)->packed_functions[fid].module = (Module *)(*webgpuModule);
        (*webgpuModule)->packed_functions[fid].exec =
            (TVMBackendPackedCFunc)TVM_RT_WASM_WebGPUWrappedFunction;
        (*webgpuModule)->packed_functions[fid].reserved = fid;
    }
}

/**
 * @brief Create a WebGPU module instance from the byte stream.
 * @param reader The module binary reader.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_WebGPUModuleCreate(ModuleBinaryReader *reader, Module **out) {
    *out = NULL;
    const char *cur_ptr;
    int status = -1;

    // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
    TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t func_map_size = (size_t) * (uint64_t *)cur_ptr;

    TVM_RT_WASM_WebGPUModuleAllocate((WebGPUModule **)out, func_map_size);
    WebGPUModule *webgpu_module = *(WebGPUModule **)out;

    WebGPUFunctionInfo *func_info_list = webgpu_module->functions;
    for (size_t fid = 0; fid < func_map_size; ++fid) {
        WebGPUFunctionInfo *info = func_info_list + fid;
        PARSE_FUNC_INFO(webgpu_module, cur_ptr, fail_label);
    }

    // parse source map <string, string>
    TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
    size_t source_map_size = (size_t) * (uint64_t *)cur_ptr;
    if (source_map_size != func_map_size) {
        TVM_RT_SET_ERROR_AND_GOTO(fail_label,
                                  "Invalid module: function size (%zu) != source size (%zu)\n",
                                  func_map_size, source_map_size);
    }

    DeviceAPI *webgpu_dev_api = NULL;
    status = TVM_RT_WASM_DeviceAPIGet(kDLWebGPU, &webgpu_dev_api);
    if (unlikely(status)) {
        goto fail_label;
    }
    // get the device
    WGPU_Device gpu_device = (WGPU_Device)webgpu_dev_api->GetStream();

    for (size_t fid = 0; fid < source_map_size; ++fid) {
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        // key: name
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        // skip name, (equal to function names)
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, name_size, fail_label);

        // key: source
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), fail_label);
        size_t src_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(cur_ptr, src_size, fail_label);
        status = WGPU_FunctionCreate(gpu_device, &func_info_list[fid].device_func, cur_ptr,
                                     src_size, NULL, 0, func_info_list[fid].num_kernel_args);
        if (unlikely(status)) {
            goto fail_label;
        }
    }

    return 0;
fail_label:
    if (*out) {
        TVM_RT_WASM_WebGPUModuleReleaseFunc(*out);
        *out = NULL;
    }
    return status;
}
