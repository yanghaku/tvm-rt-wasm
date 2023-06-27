/*!
 * \file cuda/cuda_module.c
 * \brief implement functions for cuda module
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <cuda_common.h>
#include <device/cpu_memory.h>
#include <device/device_api.h>
#include <module/function_info.h>
#include <module/module.h>

typedef struct CUDAFunctionInfo {
    /*! \brief base information */
    BASE_FUNCTION_INFO

    /*! \brief the cuda functions in cuda module */
    CUfunction cu_function;
} CUDAFunctionInfo;

/*! \brief define the cuda module derived from module */
typedef struct CUDAModule {
    MODULE_BASE_MEMBER

    /*! \brief the cuda module */
    // todo: change it to support multi-GPU
    CUmodule cu_module;
    CUDAFunctionInfo *functions;
    uint32_t num_functions;
} CUDAModule;

static int TVM_RT_WASM_CUDAWrappedFunction(TVMValue *args, const int *type_codes, int num_args,
                                           TVMValue *ret_val, const int *ret_type_codes,
                                           void *resource_handle) {
    (void)ret_val;
    (void)ret_type_codes;

    PackedFunction *pf = (PackedFunction *)resource_handle;
    int func_id = (int)pf->reserved;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    CUDAFunctionInfo *info = ((CUDAModule *)pf->module)->functions + func_id;

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

    for (uint32_t i = 0; i < c->num_functions; ++i) {
        if (c->functions[i].func_arg_index_map) {
            TVM_RT_WASM_HeapMemoryFree(c->functions[i].func_arg_index_map);
        }
        if (c->functions[i].kernel_arg_storages) {
            TVM_RT_WASM_HeapMemoryFree(c->functions[i].kernel_arg_storages);
        }
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
    (*cudaModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*cudaModule)->module_funcs_map));
    (*cudaModule)->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_func);
    (*cudaModule)->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAFunctionInfo) * num_func);
    memset((*cudaModule)->functions, 0, sizeof(CUDAFunctionInfo) * num_func);
    (*cudaModule)->num_functions = num_func;
    for (uint32_t fid = 0; fid < num_func; ++fid) {
        (*cudaModule)->packed_functions[fid].module = (Module *)(*cudaModule);
        (*cudaModule)->packed_functions[fid].exec =
            (TVMBackendPackedCFunc)TVM_RT_WASM_CUDAWrappedFunction;
        (*cudaModule)->packed_functions[fid].reserved = fid;
    }
}

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param cudaModule the out handle
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_CUDAModuleCreate(const char *resource, int resource_type, Module **out) {

    if (resource_type == MODULE_FACTORY_RESOURCE_FILE) {
        TVM_RT_NOT_IMPLEMENT(-2);
    } else if (resource_type == MODULE_FACTORY_RESOURCE_BINARY) {
        *out = NULL;
        char *blob = (char *)resource;

        // parse format
        uint32_t fmt_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // fmt_size
        if (!((fmt_size == 5 && memcmp(blob, "cubin", fmt_size) == 0) ||
              (fmt_size == 3 && memcmp(blob, "ptx", fmt_size) == 0))) {
            blob[fmt_size] = 0;
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported binary format %s", blob);
        }
        blob += fmt_size; // fmt

        // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
        uint32_t func_map_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // func_map_size
        // allocate memory for this
        TVM_RT_WASM_CUDAModuleAllocate((CUDAModule **)out, func_map_size);
        CUDAModule *cuda_module = *(CUDAModule **)out;

        char *names = blob; // for init functions from cu_module
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            CUDAFunctionInfo *info = cuda_module->functions + fid;
            PARSE_FUNC_INFO(cuda_module, fail);
        }

        // parse data
        uint32_t source_len = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // source_len

        // init cu_module
        DeviceAPI *cuda_dev_api;
        int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &cuda_dev_api);
        if (unlikely(status)) {
            goto fail;
        }
        cuda_dev_api->SetDevice(0);

        CUDA_DRIVER_CALL_OR_GOTO(cuModuleLoadData(&cuda_module->cu_module, blob), fail);

        blob += source_len;

        // load cu_functions
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // name_size

            CUDA_DRIVER_CALL_OR_GOTO(cuModuleGetFunction(&cuda_module->functions[fid].cu_function,
                                                         cuda_module->cu_module, names),
                                     fail);

            names += name_size; // name string

            // functionInfo.name
            name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t) + name_size; // name_size + name string
            names += sizeof(uint64_t);             // num_func_args
            names += cuda_module->functions[fid].num_kernel_args * sizeof(DLDataType); // arg types

            uint32_t mp_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // num_func_arg_map
            for (uint32_t i = 0; i < mp_size; ++i) {
                name_size = (uint32_t) * (uint64_t *)names;
                names += sizeof(uint64_t) + name_size; // name_size + name string
            }
        }

        return (int)(blob - resource);

    fail:
        if (*out) {
            TVM_RT_WASM_CUDAModuleReleaseFunc(*out);
            *out = NULL;
        }
        return -1;
    } else {
        TVM_RT_SET_ERROR_RETURN(-1, "Unknown resource type %d", resource_type);
    }
}
