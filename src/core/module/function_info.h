/**
 * @file module/function_info.h
 * @brief macros for parsing the FunctionInfo in module metadata.
 */

#ifndef TVM_RT_WASM_CORE_MODULE_FUNCTION_INFO_H_INCLUDE_
#define TVM_RT_WASM_CORE_MODULE_FUNCTION_INFO_H_INCLUDE_

#define BASE_FUNCTION_INFO                                                                         \
    /** @brief the argument storage for function */                                                \
    void **kernel_arg_storages;                                                                    \
    /**                                                                                            \
     * @brief the rest arguments map to thread params information                                  \
     *                                                                                             \
     * -1: NULL; [0,3): grid_dim[] (blockIdx. ; [3,6): block_dim[] (ThreadIdx.                     \
     *                                                                                             \
     */                                                                                            \
    uint32_t *func_arg_index_map;                                                                  \
    /** @brief whether use dynamic shared memory */                                                \
    uint32_t use_dyn_mem;                                                                          \
    /** @brief the number of arguments of function kernel */                                       \
    uint32_t num_kernel_args;                                                                      \
    /**                                                                                            \
     * @brief the number of the rest arguments map for every function                              \
     *                                                                                             \
     * @note for every wrapped function:                                                           \
     *  num_func_args[func_id] + num_func_arg_map[func_id] + (use_dyn_mem==1) = num_args           \
     *                                                                                             \
     *  @sa TVM_RT_WASM_CUDAWrappedFunction in cuda_module.c                                       \
     *  @sa TVM_RT_WASM_WebGPUWrappedFunction in webgpu_module.c                                   \
     */                                                                                            \
    uint32_t num_func_arg_map;

#define PARSE_FUNC_INFO(_mod, _cur_ptr, fail_label)                                                \
    do {                                                                                           \
        /* key: name */                                                                            \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, sizeof(uint64_t), fail_label);           \
        size_t name_size = (size_t) * (uint64_t *)_cur_ptr;                                        \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, name_size, fail_label);                  \
        TVM_RT_WASM_TrieInsertWithLen((_mod)->module_funcs_map, (const uint8_t *)_cur_ptr,         \
                                      name_size, (_mod)->packed_functions + fid);                  \
                                                                                                   \
        /* value: FunctionInfo{name, arg_types, launch_params_tags} */                             \
                                                                                                   \
        /* FunctionInfo.name */                                                                    \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, sizeof(uint64_t), fail_label);           \
        name_size = (size_t) * (uint64_t *)_cur_ptr;                                               \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, name_size, fail_label);                  \
                                                                                                   \
        /* num_func_args */                                                                        \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, sizeof(uint64_t), fail_label);           \
        size_t num_kernel_arg = (size_t) * (uint64_t *)_cur_ptr;                                   \
        info->num_kernel_args = num_kernel_arg;                                                    \
        info->kernel_arg_storages =                                                                \
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(void **) * (num_kernel_arg));                       \
                                                                                                   \
        /* arg types */                                                                            \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(                                                   \
            _cur_ptr, info->num_kernel_args * sizeof(DLDataType), fail_label);                     \
                                                                                                   \
        /* num_func_arg_map */                                                                     \
        TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, sizeof(uint64_t), fail_label);           \
        size_t mp_size = (size_t) * (uint64_t *)_cur_ptr;                                          \
        info->num_func_arg_map = mp_size;                                                          \
                                                                                                   \
        /* allocate memory for arg_index_map */                                                    \
        info->func_arg_index_map = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * mp_size);        \
        for (size_t i = 0; i < mp_size; ++i) {                                                     \
            TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, sizeof(uint64_t), fail_label);       \
            name_size = (size_t) * (uint64_t *)_cur_ptr;                                           \
            TVM_RT_WASM_ModuleBinaryCheckReadOrGoto(_cur_ptr, name_size, fail_label);              \
                                                                                                   \
            if (name_size == 24 &&                                                                 \
                memcmp(_cur_ptr, "tir.use_dyn_shared_memory", name_size) == 0) {                   \
                if (unlikely(i + 1 != mp_size)) {                                                  \
                    const char *msg =                                                              \
                        "binary parse error: the tir.use_dyn_shared_memory must in last!\n";       \
                    TVM_RT_SET_ERROR_AND_GOTO(fail_label, "%s", msg);                              \
                }                                                                                  \
                --info->num_func_arg_map;                                                          \
                info->use_dyn_mem = 1;                                                             \
            } else if (name_size > 17 && memcmp(_cur_ptr, "paramWriteAccess:", 17) == 0) {         \
                /* no nothing now */                                                               \
            } else if (name_size == 10 && memcmp(_cur_ptr, "blockIdx.", 9) == 0) {                 \
                info->func_arg_index_map[i] = (uint8_t)(_cur_ptr[9] - 'x');                        \
            } else if (name_size == 11 && memcmp(_cur_ptr, "threadIdx.", 10) == 0) {               \
                info->func_arg_index_map[i] = (uint8_t)(_cur_ptr[10] - 'x' + 3);                   \
            } else {                                                                               \
                TVM_RT_SET_ERROR_AND_GOTO(fail_label, "unknown params Tags: %s\n", _cur_ptr);      \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CHECK_DYN_MEM()                                                                            \
    do {                                                                                           \
        if (info->use_dyn_mem) {                                                                   \
            if (unlikely(num_kernel_args + info->num_func_arg_map + 1 != (uint32_t)num_args)) {    \
                TVM_RT_SET_ERROR_RETURN(-1, "Params number expect %d, but given %d",               \
                                        num_kernel_args + info->num_func_arg_map + 1, num_args);   \
            }                                                                                      \
            if (unlikely(*(type_codes + num_args - 1) != kTVMArgInt)) {                            \
                TVM_RT_SET_ERROR_RETURN(-1, "Expect int type for param %d", num_args - 1);         \
            }                                                                                      \
            dyn_shared_mem_size = (size_t)args[num_args - 1].v_int64;                              \
        } else {                                                                                   \
            if (unlikely(num_kernel_args + info->num_func_arg_map != (uint32_t)num_args)) {        \
                TVM_RT_SET_ERROR_RETURN(-1, "Params number expect %d, but given %d",               \
                                        num_kernel_args + info->num_func_arg_map, num_args);       \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#define CHECK_AND_GET_DIM()                                                                        \
    do {                                                                                           \
        for (uint32_t i = 0; i < info->num_func_arg_map; ++i) {                                    \
            if (unlikely(*(type_codes + i + num_kernel_args) != kTVMArgInt)) {                     \
                TVM_RT_SET_ERROR_RETURN(-1, "Expect int type for param %d", i);                    \
            }                                                                                      \
            if (info->func_arg_index_map[i] >= 3) {                                                \
                block_dim[info->func_arg_index_map[i] - 3] = args[num_kernel_args + i].v_int64;    \
            } else {                                                                               \
                grid_dim[info->func_arg_index_map[i]] = args[num_kernel_args + i].v_int64;         \
            }                                                                                      \
        }                                                                                          \
    } while (0)

#endif // TVM_RT_WASM_CORE_MODULE_FUNCTION_INFO_H_INCLUDE_
