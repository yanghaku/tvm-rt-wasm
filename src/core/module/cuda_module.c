/*!
 * \file module/cuda_module.c
 * \brief implement functions for cuda_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/device_api.h>
#include <module/cuda_module.h>
#include <module/metadata.h>
#include <stdint.h>

#if USE_CUDA // USE_CUDA = 1

struct CUDAFunctionInfo {
    /*! \brief base information */
    BASE_FUNCTION_INFO

    /*! \brief the cuda functions in cuda module */
    CUfunction cu_function;
};

static int TVM_RT_WASM_CUDAWrappedFunction(TVMValue *args, const int *type_codes, int num_args, TVMValue *ret_val,
                                           int *ret_type_codes, void *resource_handle) {
    PackedFunction *pf = (PackedFunction *)resource_handle;
    int func_id = (int)pf->reserved;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    CUDAFunctionInfo *info = ((CUDAModule *)pf->module)->functions + func_id;

    uint32_t num_kernel_args = info->num_kernel_args;
    CHECK_DYN_MEM();

    for (uint32_t i = 0; i < info->num_func_arg_map; ++i) {
        if (unlikely(*(type_codes + i + num_kernel_args) != kTVMArgInt)) {
            SET_ERROR_RETURN(-1, "params type error, expect int type");
        }
        if (info->func_arg_index_map[i] >= 3) {
            block_dim[info->func_arg_index_map[i] - 3] = args[num_kernel_args + i].v_int64;
        } else {
            grid_dim[info->func_arg_index_map[i]] = args[num_kernel_args + i].v_int64;
        }
    }

    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        info->kernel_arg_storages[i] = &args[i].v_handle;
    }

    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &deviceApi);
    if (unlikely(status)) {
        return status;
    }

    CUstream stream = (CUstream)deviceApi->GetStream();
    CUDA_DRIVER_CALL(cuLaunchKernel(info->cu_function, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
                                    block_dim[1], block_dim[2], dyn_shared_mem_size, stream, info->kernel_arg_storages,
                                    NULL));

    return status;
}

static int TVM_RT_WASM_CUDAModuleReleaseFunc(Module *self) {
    CUDAModule *c = (CUDAModule *)self;
    MODULE_BASE_MEMBER_FREE(c);

    for (uint32_t i = 0; i < c->num_functions; ++i) {
        TVM_RT_WASM_HeapMemoryFree(c->functions[i].func_arg_index_map);
        TVM_RT_WASM_HeapMemoryFree(c->functions[i].kernel_arg_storages);
    }
    TVM_RT_WASM_HeapMemoryFree(c->functions);
    TVM_RT_WASM_HeapMemoryFree(c->packed_functions);

    cuModuleUnload(c->cu_module);
    // free self
    TVM_RT_WASM_HeapMemoryFree(c);
    return 0;
}

static void TVM_RT_WASM_CUDAModuleAllocate(CUDAModule **cudaModule, uint32_t num_func) {
    *cudaModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAModule));
    memset(*cudaModule, 0, sizeof(CUDAModule));
    (*cudaModule)->Release = TVM_RT_WASM_CUDAModuleReleaseFunc;
    TVM_RT_WASM_TrieCreate(&((*cudaModule)->module_funcs_map));
    (*cudaModule)->packed_functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_func);
    (*cudaModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAFunctionInfo) * num_func);
    (*cudaModule)->num_functions = num_func;
    for (uint32_t fid = 0; fid < num_func; ++fid) {
        (*cudaModule)->packed_functions[fid].module = (*cudaModule);
        (*cudaModule)->packed_functions[fid].exec = (TVMBackendPackedCFunc)TVM_RT_WASM_CUDAWrappedFunction;
        (*cudaModule)->packed_functions[fid].reserved = fid;
    }
}

#endif // USE_CUDA

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param cudaModule the out handle
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_CUDAModuleCreate(const char *resource, int resource_type, CUDAModule **cudaModule) {
#if USE_CUDA // USE_CUDA = 1

    if (resource_type == MODULE_FACTORY_RESOURCE_FILE) {
        SET_ERROR_RETURN(-1, "creating from file is unsupported yet");
    } else if (resource_type == MODULE_FACTORY_RESOURCE_BINARY) {
        char *blob = (char *)resource;

        // parse format
        uint32_t fmt_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // fmt_size
        if (!((fmt_size == 5 && memcmp(blob, "cubin", fmt_size) == 0) ||
              (fmt_size == 3 && memcmp(blob, "ptx", fmt_size) == 0))) {
            blob[fmt_size] = 0;
            SET_ERROR_RETURN(-1, "unsupported binary format %s\n", blob);
        }
        blob += fmt_size; // fmt

        // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
        uint32_t func_map_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // func_map_size
        // allocate memory for this
        TVM_RT_WASM_CUDAModuleAllocate(cudaModule, func_map_size);
        char *names = blob; // for init functions from cu_module
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            CUDAFunctionInfo *info = (*cudaModule)->functions + fid;
            PARSE_FUNC_INFO(cudaModule);
        }

        // parse data
        uint32_t source_len = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // source_len

        // init cu_module
        DeviceAPI *cuda_dev_api;
        int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &cuda_dev_api);
        if (unlikely(status)) {
            return status;
        }
        cuda_dev_api->SetDevice(0);

        CUDA_DRIVER_CALL(cuModuleLoadData(&(*cudaModule)->cu_module, blob));

        blob += source_len;

        // load cu_functions
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // name_size

            CUDA_DRIVER_CALL(
                cuModuleGetFunction(&(*cudaModule)->functions[fid].cu_function, (*cudaModule)->cu_module, names));

            names += name_size; // name string

            // functionInfo.name
            name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t) + name_size;                                       // name_size + name string
            names += sizeof(uint64_t);                                                   // num_func_args
            names += (*cudaModule)->functions[fid].num_kernel_args * sizeof(DLDataType); // arg types

            uint32_t mp_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // num_func_arg_map
            for (uint32_t i = 0; i < mp_size; ++i) {
                name_size = (uint32_t) * (uint64_t *)names;
                names += sizeof(uint64_t) + name_size; // name_size + name string
            }
        }

        return (int)(blob - resource);
    } else {
        SET_ERROR_RETURN(-1, "unknown resource type %d\n", resource_type);
    }

#else
    CUDA_NOT_SUPPORTED();
#endif
}
