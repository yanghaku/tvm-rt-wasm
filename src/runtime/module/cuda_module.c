/*!
 * \file src/runtime/module/cuda_module.c
 * \brief implement functions for cuda_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdint.h>
#include <tvm/runtime/device/device_api.h>
#include <tvm/runtime/module/cuda_module.h>

static CUDAModule *g_module = NULL;

static int cudaWrappedFunction(TVMValue *args, const int *type_codes, int num_args, TVMValue *ret_val,
                               int *ret_type_codes, void *resource_handle) {
    int func_id = (int)*((uintptr_t *)resource_handle);
    size_t block_dim[] = {1, 1, 1};
    size_t grid_dim[] = {1, 1, 1};
    size_t dyn_shared_mem_size = 0;
    CUDAModule *module = g_module;

    uint32_t num_kernel_arg = module->num_kernel_args[func_id];
    if (module->use_dyn_mem[func_id]) {
        if (unlikely(num_kernel_arg + module->num_func_arg_map[func_id] + 1 != (uint32_t)num_args)) {
            SET_ERROR_RETURN(-1, "params number error, expect %d, but given %d\n",
                             num_kernel_arg + module->num_func_arg_map[func_id] + 1, num_args);
        }
        if (unlikely(*(type_codes + num_args - 1) != kTVMArgInt)) {
            SET_ERROR_RETURN(-1, "params type error: expect int type to use dynamic shared memory");
        }
        dyn_shared_mem_size = (size_t)args[num_args - 1].v_int64;
    } else {
        if (unlikely(num_kernel_arg + module->num_func_arg_map[func_id] != (uint32_t)num_args)) {
            SET_ERROR_RETURN(-1, "params number error, expect %d, but given %d\n",
                             num_kernel_arg + module->num_func_arg_map[func_id], num_args);
        }
    }

    for (uint32_t i = 0; i < module->num_func_arg_map[func_id]; ++i) {
        if (unlikely(*(type_codes + i + num_kernel_arg) != kTVMArgInt)) {
            SET_ERROR_RETURN(-1, "params type error, expect int type");
        }
        if (module->func_arg_index_map[func_id][i] >= 3) {
            block_dim[module->func_arg_index_map[func_id][i] - 3] = args[num_kernel_arg + i].v_int64;
        } else {
            grid_dim[module->func_arg_index_map[func_id][i]] = args[num_kernel_arg + i].v_int64;
        }
    }

    for (uint32_t i = 0; i < num_kernel_arg; ++i) {
        module->kernel_arg_storages[func_id][i] = &args[i].v_handle;
    }
    module->kernel_arg_storages[func_id][num_kernel_arg] = NULL;

    DeviceAPI *deviceApi;
    int status = DeviceAPIGet(kDLCUDA, &deviceApi);
    if (unlikely(status)) {
        return status;
    }

    CUstream stream = (CUstream)deviceApi->GetStream();
    CUDA_DRIVER_CALL(cuLaunchKernel(module->functions[func_id], grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
                                    block_dim[1], block_dim[2], dyn_shared_mem_size, stream,
                                    module->kernel_arg_storages[func_id], NULL));

    return status;
}

static int CUDAModuleReleaseFunc(Module *self) {
    DLDevice cpu = {kDLCPU, 0};
    CUDAModule *c = (CUDAModule *)self;

    if (c->imports) {
        for (uint32_t i = 0; i < c->num_imports; ++i) {
            c->imports[i]->Release(c->imports[i]);
        }
        TVMDeviceFreeDataSpace(cpu, c->imports);
    }
    if (c->env_funcs_map) {
        TrieRelease(c->env_funcs_map);
    }
    if (c->module_funcs_map) {
        TrieRelease(c->module_funcs_map);
    }
    for (uint32_t i = 0; i < c->num_functions; ++i) {
        TVMDeviceFreeDataSpace(cpu, c->func_arg_index_map[i]);
        TVMDeviceFreeDataSpace(cpu, c->kernel_arg_storages[i]);
    }
    TVMDeviceFreeDataSpace(cpu, c->func_arg_index_map);
    TVMDeviceFreeDataSpace(cpu, c->use_dyn_mem);
    TVMDeviceFreeDataSpace(cpu, c->functions);
    TVMDeviceFreeDataSpace(cpu, c->kernel_arg_storages);
    TVMDeviceFreeDataSpace(cpu, c->num_kernel_args);
    TVMDeviceFreeDataSpace(cpu, c->num_func_arg_map);
    CUDA_DRIVER_CALL(cuModuleUnload(c->cu_module));
    // free self
    return TVMDeviceFreeDataSpace(cpu, c);
}

static void CUDAModuleAllocate(CUDAModule **cudaModule, uint32_t num_func) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(CUDAModule), 0, no_type, (void **)cudaModule);
    memset(*cudaModule, 0, sizeof(CUDAModule));
    (*cudaModule)->Release = CUDAModuleReleaseFunc;
    TrieCreate(&(*cudaModule)->module_funcs_map);
    TVMDeviceAllocDataSpace(cpu, sizeof(CUfunction) * num_func, 0, no_type, (void **)&(*cudaModule)->functions);
    TVMDeviceAllocDataSpace(cpu, sizeof(uint32_t) * num_func, 0, no_type, (void **)&(*cudaModule)->num_kernel_args);
    TVMDeviceAllocDataSpace(cpu, sizeof(uint32_t) * num_func, 0, no_type, (void **)&(*cudaModule)->num_func_arg_map);
    TVMDeviceAllocDataSpace(cpu, sizeof(uint8_t) * num_func, 0, no_type, (void **)&(*cudaModule)->use_dyn_mem);
    TVMDeviceAllocDataSpace(cpu, sizeof(void **) * num_func, 0, no_type, (void **)&(*cudaModule)->kernel_arg_storages);
    TVMDeviceAllocDataSpace(cpu, sizeof(char *) * num_func, 0, no_type, (void **)&(*cudaModule)->func_arg_index_map);
    (*cudaModule)->num_functions = num_func;
}

/*!
 * \brief create a cuda module instance from file or binary
 * @param resource the file name or binary pointer
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param cudaModule the out handle
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int CUDAModuleCreate(const char *resource, int resource_type, CUDAModule **cudaModule) {
#if USE_CUDA // USE_CUDA = 1

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};

    if (resource_type == MODULE_FACTORY_RESOURCE_FILE) {
        SET_ERROR_RETURN(-1, "creating from file is unsupported yet");
    } else if (resource_type == MODULE_FACTORY_RESOURCE_BINARY) {
        char *blob = (char *)resource;

        // parse format
        uint32_t fmt_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // fmt_size
        if (memcmp(blob, "ptx", fmt_size) != 0) {
            blob[fmt_size] = 0;
            SET_ERROR_RETURN(-1, "unsupported binary format %s\n", blob);
        }
        blob += fmt_size; // fmt

        // parse function map: <string, FunctionInfo{name, arg_types, launch_params_tags} >
        uint32_t func_map_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t); // func_map_size
        // allocate memory for this
        CUDAModuleAllocate(cudaModule, func_map_size);
        char *names = blob; // for init cu_functions from cu_module
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t); // name_size

            char byte = blob[name_size];
            blob[name_size] = 0; // the end of string must be '\0'
            // encode this function as TVMFunctionHandle and insert to map
            TVMFunctionHandle handle = TVM_FUNCTION_HANDLE_ENCODE(cudaWrappedFunction, fid);
            TrieInsert((*cudaModule)->module_funcs_map, (const uint8_t *)blob, handle);
            blob += name_size; // name string
            *blob = byte;      // back this byte

            // value: FunctionInfo{name, arg_types, launch_params_tags}
            name_size = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t) + name_size; // name_size + name string

            uint32_t num_kernel_arg = (uint32_t) * (uint64_t *)blob;
            (*cudaModule)->num_kernel_args[fid] = num_kernel_arg;
            TVMDeviceAllocDataSpace(cpu, sizeof(void **) * num_kernel_arg, 0, no_type,
                                    (void **)&(*cudaModule)->kernel_arg_storages[fid]);

            blob += sizeof(uint64_t);                                         // num_func_args
            blob += (*cudaModule)->num_kernel_args[fid] * sizeof(DLDataType); // arg types

            uint32_t mp_size = (uint32_t) * (uint64_t *)blob;
            (*cudaModule)->num_func_arg_map[fid] = mp_size;
            blob += sizeof(uint64_t); // num_func_arg_map

            // allocate memory for arg_index_map
            TVMDeviceAllocDataSpace(cpu, sizeof(uint8_t) * mp_size, 0, no_type,
                                    (void **)&(*cudaModule)->func_arg_index_map[fid]);
            for (uint32_t i = 0; i < mp_size; ++i) {
                name_size = (uint32_t) * (uint64_t *)blob;
                blob += sizeof(uint64_t); // name_size

                if (name_size == 24 && memcmp(blob, "tir.use_dyn_shared_memory", name_size) == 0) {
                    if (unlikely(i + 1 != mp_size)) {
                        const char *msg = "cuda binary parse error: the tir.use_dyn_shared_memory must in last!\n";
                        fprintf(stderr, "%s", msg);
                        SET_ERROR_RETURN(-1, "%s", msg);
                    }
                    --(*cudaModule)->num_func_arg_map[fid];
                    (*cudaModule)->use_dyn_mem[fid] = 1;
                } else if (name_size == 10 && memcmp(blob, "blockIdx.", 9) == 0) {
                    (*cudaModule)->func_arg_index_map[fid][i] = (uint8_t)(blob[9] - 'x');
                } else if (name_size == 11 && memcmp(blob, "threadIdx.", 10) == 0) {
                    (*cudaModule)->func_arg_index_map[fid][i] = (uint8_t)(blob[10] - 'x' + 3);
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

        char byte = blob[source_len]; // backup the byte
        blob[source_len] = 0;         // the end of string must be '\0'
        // init cu_module
        DeviceAPI *cuda_dev_api;
        int status = DeviceAPIGet(kDLCUDA, &cuda_dev_api);
        if (unlikely(status)) {
            return status;
        }
        cuda_dev_api->SetDevice(0);
        CUDA_DRIVER_CALL(cuModuleLoadData(&(*cudaModule)->cu_module, blob));
        blob += source_len;
        *blob = byte; // restore it

        // load cu_functions
        for (uint32_t fid = 0; fid < func_map_size; ++fid) {
            // key: name
            uint32_t name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // name_size
            byte = names[name_size];   // backup the last byte
            names[name_size] = 0;      // the end of string must be '\0'

            CUDA_DRIVER_CALL(cuModuleGetFunction((*cudaModule)->functions + fid, (*cudaModule)->cu_module, names));

            names += name_size; // name string
            *names = byte;      // restore this byte

            // functionInfo.name
            name_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t) + name_size;                             // name_size + name string
            names += sizeof(uint64_t);                                         // num_func_args
            names += (*cudaModule)->num_kernel_args[fid] * sizeof(DLDataType); // arg types

            uint32_t mp_size = (uint32_t) * (uint64_t *)names;
            names += sizeof(uint64_t); // num_func_arg_map
            for (uint32_t i = 0; i < mp_size; ++i) {
                name_size = (uint32_t) * (uint64_t *)names;
                names += sizeof(uint64_t) + name_size; // name_size + name string
            }
        }

        g_module = *cudaModule;
        return blob - resource;
    } else {
        SET_ERROR_RETURN(-1, "unknown resource type %d\n", resource_type);
    }

#else
    CUDA_NOT_SUPPORTED();
#endif
}
