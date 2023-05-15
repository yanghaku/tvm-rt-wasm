/*!
 * \file module/webgpu_module.c
 * \brief implement functions for webgpu_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/device_api.h>
#include <module/metadata.h>
#include <module/webgpu_module.h>
#include <stdint.h>
#include <utils/webgpu_common.h>

#if USE_WEBGPU // USE_WEBGPU = 1

struct WebGPUFunctionInfo {
    BASE_FUNCTION_INFO

    WGPU_Function device_func;
};

static int TVM_RT_WASM_WebGPUWrappedFunction(TVMValue *args, const int *type_codes, int num_args, TVMValue *ret_val,
                                             int *ret_type_codes, void *resource_handle) {
    PackedFunction *pf = (PackedFunction *)resource_handle;
    int func_id = (int)pf->reserved;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    WebGPUFunctionInfo *info = ((WebGPUModule *)pf->module)->functions + func_id;

    uint32_t num_kernel_args = info->num_kernel_args;
    CHECK_DYN_MEM();
    if (dyn_shared_mem_size != 0) {
        SET_ERROR_RETURN(-1, "WebGPU cannot support dynamic shared memory, but got size %d.", dyn_shared_mem_size);
    }

    CHECK_AND_GET_DIM();
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        info->kernel_arg_storages[i] = args[i].v_handle;
    }

    int status = WGPU_FunctionRun(info->device_func, grid_dim, block_dim, (WGPU_Memory *)info->kernel_arg_storages,
                                  num_kernel_args);

    return status;
}

static int TVM_RT_WASM_WebGPUModuleReleaseFunc(Module *self) {
    WebGPUModule *w = (WebGPUModule *)self;
    MODULE_BASE_MEMBER_FREE(w);

    for (uint32_t i = 0; i < w->num_functions; ++i) {
        TVM_RT_WASM_HeapMemoryFree(w->functions[i].func_arg_index_map);
        TVM_RT_WASM_HeapMemoryFree(w->functions[i].kernel_arg_storages);
        WGPU_FunctionFree(w->functions[i].device_func);
    }
    TVM_RT_WASM_HeapMemoryFree(w->functions);
    TVM_RT_WASM_HeapMemoryFree(w->packed_functions);

    // free self
    TVM_RT_WASM_HeapMemoryFree(w);
    return 0;
}

static void TVM_RT_WASM_WebGPUModuleAllocate(WebGPUModule **webgpuModule, uint32_t num_func) {
    *webgpuModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUModule));
    memset(*webgpuModule, 0, sizeof(WebGPUModule));
    (*webgpuModule)->Release = TVM_RT_WASM_WebGPUModuleReleaseFunc;
    TVM_RT_WASM_TrieCreate(&((*webgpuModule)->module_funcs_map));
    (*webgpuModule)->packed_functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_func);
    (*webgpuModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(WebGPUFunctionInfo) * num_func);
    (*webgpuModule)->num_functions = num_func;
    for (uint32_t fid = 0; fid < num_func; ++fid) {
        (*webgpuModule)->packed_functions[fid].module = (*webgpuModule);
        (*webgpuModule)->packed_functions[fid].exec = (TVMBackendPackedCFunc)TVM_RT_WASM_WebGPUWrappedFunction;
        (*webgpuModule)->packed_functions[fid].reserved = fid;
    }
}

#endif // USE_WEBGPU

/*!
 * \brief create a webgpu module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param webgpuModule the out handle
 * @return >=0 if successful  (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_WebGPUModuleCreate(const char *resource, int resource_type, WebGPUModule **webGPUModule) {
#if USE_WEBGPU // USE_WEBGPU = 1

    if (resource_type == MODULE_FACTORY_RESOURCE_FILE) {
        SET_ERROR_RETURN(-1, "creating from file is unsupported yet");
    } else if (resource_type == MODULE_FACTORY_RESOURCE_BINARY) {
        char *blob = (char *)resource;

        // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
        uint32_t func_map_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // func_map_size

        TVM_RT_WASM_WebGPUModuleAllocate(webGPUModule, func_map_size);
        char *names = blob;

        WebGPUFunctionInfo *func_info_list = (*webGPUModule)->functions;
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            WebGPUFunctionInfo *info = func_info_list + fid;
            PARSE_FUNC_INFO(webGPUModule);
        }

        // parse source map <string, string>
        uint32_t source_map_size = (uint32_t) * (uint64_t *)blob;
        if (source_map_size != func_map_size) {
            SET_ERROR_RETURN(-1, "Invalid module: function size (%d) != source size (%d)\n", func_map_size,
                             source_map_size);
        }

        DeviceAPI *webgpu_dev_api = NULL;
        int status = TVM_RT_WASM_DeviceAPIGet(kDLWebGPU, &webgpu_dev_api);
        if (unlikely(status)) {
            return status;
        }
        // get the device
        WGPU_Device gpu_device = (WGPU_Device)webgpu_dev_api->GetStream();

        blob += sizeof(uint64_t); // source_map_size
        for (uint32_t fid = 0; fid < source_map_size; ++fid) {
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t); // name_size
            blob += name_size;        // name string
            // skip name, (equal to function names)

            // key: source
            uint32_t src_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t); // src_size

            WGPU_CALL(WGPU_FunctionCreate(gpu_device, &func_info_list[fid].device_func, blob, src_size,
                                          func_info_list[fid].num_kernel_args));

            blob += src_size; // source string
        }

        return (int)(blob - resource);
    } else {
        SET_ERROR_RETURN(-1, "unknown resource type %d\n", resource_type);
    }
#else
    WebGPU_NOT_SUPPORTED();
#endif // USE_WEBGPU
}
