/*!
 * \file module/module.c
 * \brief implement functions for module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <module/module.h>
#include <string.h>

#define MODULE_CREATE_IF_NO_SUPPORT(dev)                                                           \
    _Pragma(TOSTRING(weak TVM_RT_WASM_##dev##ModuleCreate));                                       \
    int TVM_RT_WASM_##dev##ModuleCreate(const char *resource, int resource_type, Module **out) {   \
        (void)resource;                                                                            \
        (void)resource_type;                                                                       \
        *out = NULL;                                                                               \
        TVM_RT_##dev##_NOT_LINK();                                                                 \
        return -1;                                                                                 \
    }

MODULE_CREATE_IF_NO_SUPPORT(CUDA)
MODULE_CREATE_IF_NO_SUPPORT(WebGPU)

/*! \brief Default function for module get function. */
int TVM_RT_WASM_DefaultModuleGetFunction(Module *mod, const char *func_name, int query_imports,
                                         TVMFunctionHandle *out) {
    int status = TVM_RT_WASM_TrieQuery(mod->module_funcs_map, (const uint8_t *)func_name, out);
    if (likely(status != TRIE_NOT_FOUND)) {
        return status;
    }

    if (query_imports) {
        status = TVM_RT_WASM_TrieQuery(mod->env_funcs_map, (const uint8_t *)func_name, out);
    }
    return status;
}

/*!
 * \brief Load from binary blob
 * @param blob the dev_blob binary
 * @param lib_module the root module handle
 * @return 0 if successful
 */
int TVM_RT_WASM_ModuleLoadBinaryBlob(const char *blob, Module **lib_module) {
    //    uint64_t blob_size = *(uint64_t *)blob;
    blob += sizeof(uint64_t);

    uint32_t key_num = (uint32_t) * (uint64_t *)blob;
    blob += sizeof(uint64_t);

    Module **modules = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * key_num);
    uint64_t *import_tree_row_ptr = NULL;
    uint64_t *import_tree_child_indices = NULL;
    uint32_t num_modules = 0;
    uint32_t num_import_tree_row_ptr = 0;
    uint32_t num_import_tree_child_indices = 0;
    int status;

    for (uint32_t i = 0; i < key_num; ++i) {
        uint32_t mod_type_key_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t);

        if (mod_type_key_size == 4 && !memcmp(blob, "_lib", mod_type_key_size)) {
            modules[num_modules++] = *lib_module;
            blob += mod_type_key_size;
        } else if (mod_type_key_size == 12 && !memcmp(blob, "_import_tree", mod_type_key_size)) {
            blob += mod_type_key_size;

            num_import_tree_row_ptr = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t);
            if (import_tree_row_ptr == NULL) {
                import_tree_row_ptr =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint64_t) * num_import_tree_row_ptr);
                memcpy(import_tree_row_ptr, blob, sizeof(uint64_t) * num_import_tree_row_ptr);
            }
            blob += sizeof(uint64_t) * num_import_tree_row_ptr;

            num_import_tree_child_indices = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t);
            if (import_tree_child_indices == NULL) {
                import_tree_child_indices =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint64_t) * num_import_tree_child_indices);
                memcpy(import_tree_child_indices, blob,
                       sizeof(uint64_t) * num_import_tree_child_indices);
            }
            blob += sizeof(uint64_t) * num_import_tree_child_indices;

        } else if (mod_type_key_size == 15 && !memcmp(blob, "metadata_module", mod_type_key_size)) {
            blob += mod_type_key_size; // empty module
            modules[num_modules++] = NULL;
        } else {
            const char *key = blob;
            blob += mod_type_key_size;
            status = TVM_RT_WASM_ModuleFactory(key, blob, MODULE_FACTORY_RESOURCE_BINARY,
                                               &modules[num_modules]);
            if (unlikely(status <= 0)) { // ModuleFactory will return offset
                if (modules) {
                    TVM_RT_WASM_HeapMemoryFree(modules);
                }
                if (import_tree_row_ptr) {
                    TVM_RT_WASM_HeapMemoryFree(import_tree_row_ptr);
                }
                if (import_tree_child_indices) {
                    TVM_RT_WASM_HeapMemoryFree(import_tree_child_indices);
                }
                return -1;
            }
            blob += status;
            ++num_modules;
        }
    }

    if (import_tree_row_ptr == NULL) { // no _import_tree, will no _lib
        (*lib_module)->allocated_imports_size = key_num;
        (*lib_module)->imports = modules;
        (*lib_module)->num_imports = num_modules;
        // cache all env function
        for (uint32_t i = 0; i < num_modules; ++i) {
            if (modules[i]->module_funcs_map) {
                TVM_RT_WASM_TrieInsertAll((*lib_module)->env_funcs_map,
                                          modules[i]->module_funcs_map);
            }
        }
    } else {
        for (uint32_t i = 0; i < num_modules; ++i) {
            if (unlikely(i + 1 >= num_import_tree_row_ptr ||
                         import_tree_row_ptr[i] > import_tree_row_ptr[i + 1])) {
                break;
            }

            uint32_t num_imports = (uint32_t)(import_tree_row_ptr[i + 1] - import_tree_row_ptr[i]);
            if (modules[i] == NULL) { // empty module, such as metadata_module
                if (num_imports == 1) {
                    modules[i] = modules[import_tree_child_indices[import_tree_row_ptr[i]]];
                    continue;
                } else {
                    // todo
                    continue;
                }
            }
            modules[i]->num_imports = num_imports;
            modules[i]->allocated_imports_size = num_imports;
            if (num_imports == 0) {
                continue;
            }
            modules[i]->imports = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * num_imports);
            memset(modules[i]->imports, 0, sizeof(Module *) * num_imports);

            for (uint32_t j = import_tree_row_ptr[i], x = 0; j < import_tree_row_ptr[i + 1];
                 ++j, x++) {
                if (unlikely(j >= num_import_tree_child_indices)) {
                    break;
                }
                modules[i]->imports[x] = modules[import_tree_child_indices[j]];
            }
        }
        // lib_module will be the root in import tree
        *lib_module = modules[0];
        for (uint32_t i = 1; i < num_modules; ++i) {
            if (modules[i]->module_funcs_map) {
                TVM_RT_WASM_TrieInsertAll((*lib_module)->env_funcs_map,
                                          modules[i]->module_funcs_map);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(modules);
        TVM_RT_WASM_HeapMemoryFree(import_tree_row_ptr);
        TVM_RT_WASM_HeapMemoryFree(import_tree_child_indices);
    }
    return 0;
}

/*!
 * \brief create a module instance for given type
 * @param type the module type or file format
 * @param resource filename or binary source
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param out the pointer to receive created instance
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int TVM_RT_WASM_ModuleFactory(const char *type, const char *resource, int resource_type,
                              Module **out) {
    if (!memcmp(type, MODULE_SYSTEM_LIB, strlen(MODULE_SYSTEM_LIB))) {
        return TVM_RT_WASM_SystemLibraryModuleCreate(out);
    }
    if (!memcmp(type, "so", 2) || !memcmp(type, "dll", 3) || !memcmp(type, "dylib", 5)) {
        if (unlikely(resource_type != MODULE_FACTORY_RESOURCE_FILE)) {
            TVM_RT_SET_ERROR_RETURN(-1, "The dso library can only be load from file");
        }
        return TVM_RT_WASM_DSOLibraryModuleCreate(resource, out);
    }
    if (!memcmp(type, "cuda", 4)) {
        return TVM_RT_WASM_CUDAModuleCreate(resource, resource_type, out);
    }
    if (!memcmp(type, "webgpu", 6)) {
        return TVM_RT_WASM_WebGPUModuleCreate(resource, resource_type, out);
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported module type %s", type);
}
