/*!
 * \file src/runtime/module/cuda_module.c
 * \brief implement functions for cuda_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdint.h>
#include <tvm/runtime/device/device_api.h>
#include <tvm/runtime/module/cuda_module.h>

#if USE_CUDA // USE_CUDA = 1

struct CUDAFunctionInfo {
    /*! \brief the cuda functions in cuda module */
    CUfunction cu_function;
    /*! \brief the argument storage for function */
    void **kernel_arg_storages;
    /*!
     * \brief the rest arguments map to thread params information
     *
     * -1: NULL; [0,3): grid_dim[] (blockIdx. ; [3,6): block_dim[] (ThreadIdx.
     *
     */
    uint32_t *func_arg_index_map;
    /*! \brief whether use dynamic shared memory */
    uint32_t use_dyn_mem;
    /*! \brief the number of arguments of function kernel */
    uint32_t num_kernel_args;
    /*!
     * \brief the number of the rest arguments map for every function
     *
     * \note for every wrapped function:
     *  num_func_args[func_id] + num_func_arg_map[func_id] + (use_dyn_mem==1) = num_args
     *
     *  \sa TVM_RT_WASM_CUDAWrappedFunction in cuda_module.c
     */
    uint32_t num_func_arg_map;
    /*! \brief the number of functions */
};

static int TVM_RT_WASM_CUDAWrappedFunction(TVMValue *args, const int *type_codes, int num_args, TVMValue *ret_val,
                                           int *ret_type_codes, void *resource_handle) {
    PackedFunction *pf = (PackedFunction *)resource_handle;
    int func_id = (int)pf->reserved;
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    CUDAFunctionInfo *info = ((CUDAModule *)pf->module)->functions + func_id;

    uint32_t num_kernel_arg = info->num_kernel_args;
    if (info->use_dyn_mem) {
        if (unlikely(num_kernel_arg + info->num_func_arg_map + 1 != (uint32_t)num_args)) {
            SET_ERROR_RETURN(-1, "params number error, expect %d, but given %d\n",
                             num_kernel_arg + info->num_func_arg_map + 1, num_args);
        }
        if (unlikely(*(type_codes + num_args - 1) != kTVMArgInt)) {
            SET_ERROR_RETURN(-1, "params type error: expect int type to use dynamic shared memory");
        }
        dyn_shared_mem_size = (size_t)args[num_args - 1].v_int64;
    } else {
        if (unlikely(num_kernel_arg + info->num_func_arg_map != (uint32_t)num_args)) {
            SET_ERROR_RETURN(-1, "params number error, expect %d, but given %d\n",
                             num_kernel_arg + info->num_func_arg_map, num_args);
        }
    }

    for (uint32_t i = 0; i < info->num_func_arg_map; ++i) {
        if (unlikely(*(type_codes + i + num_kernel_arg) != kTVMArgInt)) {
            SET_ERROR_RETURN(-1, "params type error, expect int type");
        }
        if (info->func_arg_index_map[i] >= 3) {
            block_dim[info->func_arg_index_map[i] - 3] = args[num_kernel_arg + i].v_int64;
        } else {
            grid_dim[info->func_arg_index_map[i]] = args[num_kernel_arg + i].v_int64;
        }
    }

    for (uint32_t i = 0; i < num_kernel_arg; ++i) {
        info->kernel_arg_storages[i] = &args[i].v_handle;
    }
    info->kernel_arg_storages[num_kernel_arg] = NULL;

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

    if (c->imports) {
        for (uint32_t i = 0; i < c->num_imports; ++i) {
            c->imports[i]->Release(c->imports[i]);
        }
        TVM_RT_WASM_HeapMemoryFree(c->imports);
    }
    if (c->env_funcs_map) {
        TVM_RT_WASM_TrieRelease(c->env_funcs_map);
    }
    if (c->module_funcs_map) {
        TVM_RT_WASM_TrieRelease(c->module_funcs_map);
    }
    for (uint32_t i = 0; i < c->num_functions; ++i) {
        TVM_RT_WASM_HeapMemoryFree(c->functions[i].func_arg_index_map);
        TVM_RT_WASM_HeapMemoryFree(c->functions[i].kernel_arg_storages);
    }
    TVM_RT_WASM_HeapMemoryFree(c->functions);
    TVM_RT_WASM_HeapMemoryFree(c->packed_functions);

    CUDA_DRIVER_CALL(cuModuleUnload(c->cu_module));
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
        (*cudaModule)->packed_functions[fid].exec = TVM_RT_WASM_CUDAWrappedFunction;
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
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t); // name_size

            TVM_RT_WASM_TrieInsertWithLen((*cudaModule)->module_funcs_map, (const uint8_t *)blob, name_size,
                                          (*cudaModule)->packed_functions + fid);
            blob += name_size; // name string

            // value: FunctionInfo{name, arg_types, launch_params_tags}
            name_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t) + name_size; // name_size + name string

            uint32_t num_kernel_arg = (uint32_t) * (uint64_t *)blob;
            info->num_kernel_args = num_kernel_arg;
            info->kernel_arg_storages = TVM_RT_WASM_HeapMemoryAlloc(sizeof(void **) * (num_kernel_arg + 1));

            blob += sizeof(uint64_t);                           // num_func_args
            blob += info->num_kernel_args * sizeof(DLDataType); // arg types

            uint32_t mp_size = (uint32_t) * (uint64_t *)blob;
            info->num_func_arg_map = mp_size;
            blob += sizeof(uint64_t); // num_func_arg_map

            // allocate memory for arg_index_map
            info->func_arg_index_map = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * mp_size);
            for (uint32_t i = 0; i < mp_size; ++i) {
                name_size = (uint32_t) * (uint64_t *)blob;
                blob += sizeof(uint64_t); // name_size

                if (name_size == 24 && memcmp(blob, "tir.use_dyn_shared_memory", name_size) == 0) {
                    if (unlikely(i + 1 != mp_size)) {
                        const char *msg = "cuda binary parse error: the tir.use_dyn_shared_memory must in last!\n";
                        fprintf(stderr, "%s", msg);
                        SET_ERROR_RETURN(-1, "%s", msg);
                    }
                    --info->num_func_arg_map;
                    info->use_dyn_mem = 1;
                } else if (name_size == 10 && memcmp(blob, "blockIdx.", 9) == 0) {
                    info->func_arg_index_map[i] = (uint8_t)(blob[9] - 'x');
                } else if (name_size == 11 && memcmp(blob, "threadIdx.", 10) == 0) {
                    info->func_arg_index_map[i] = (uint8_t)(blob[10] - 'x' + 3);
                } else {
                    blob[name_size] = '\0';
                    SET_ERROR_RETURN(-1, "unknown params Tags: %s\n", blob);
                }

                blob += name_size; // name string
            }
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

        return blob - resource;
    } else {
        SET_ERROR_RETURN(-1, "unknown resource type %d\n", resource_type);
    }

#else
    CUDA_NOT_SUPPORTED();
#endif
}
