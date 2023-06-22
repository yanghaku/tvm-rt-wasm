/*!
 * \file module/module.c
 * \brief implement functions for module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>
#include <tvm/runtime/c_backend_api.h>

#include <device/cpu_memory.h>
#include <module/cuda_module.h>
#include <module/dso_module.h>
#include <module/webgpu_module.h>

/*!
 * \brief the symbols for system library
 * \note this Trie will be managed by sys_lib_module instance after init
 */
static Trie *system_lib_symbol = NULL;
static uint32_t num_sys_lib_symbol = 0;

/*! \brief the system library module is a single instance */
static Module *sys_lib_module = NULL;

static __attribute__((destructor)) void tvm_runtime_for_webassembly_destructor_for_sys_symbol() {
    if (sys_lib_module) {
        sys_lib_module->Release(sys_lib_module);
    }

    if (system_lib_symbol) {
        TVM_RT_WASM_TrieRelease(system_lib_symbol);
    }
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
        TVM_RT_WASM_TrieCreate(&system_lib_symbol);
    }
    ++num_sys_lib_symbol;
    return TVM_RT_WASM_TrieInsert(system_lib_symbol, (const uint8_t *)name, ptr);
}

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
            import_tree_row_ptr = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint64_t) * num_import_tree_row_ptr);
            memcpy(import_tree_row_ptr, blob, sizeof(uint64_t) * num_import_tree_row_ptr);
            blob += sizeof(uint64_t) * num_import_tree_row_ptr;

            num_import_tree_child_indices = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t);
            import_tree_child_indices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint64_t) * num_import_tree_child_indices);
            memcpy(import_tree_child_indices, blob, sizeof(uint64_t) * num_import_tree_child_indices);
            blob += sizeof(uint64_t) * num_import_tree_child_indices;

        } else {
            const char *key = blob;
            blob += mod_type_key_size;
            status = TVM_RT_WASM_ModuleFactory(key, blob, MODULE_FACTORY_RESOURCE_BINARY, &modules[num_modules]);
            if (unlikely(status <= 0)) { // ModuleFactory will return offset
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
                TVM_RT_WASM_TrieInsertAll((*lib_module)->env_funcs_map, modules[i]->module_funcs_map);
            }
        }
    } else {
        for (uint32_t i = 0; i < num_modules; ++i) {
            if (unlikely(i + 1 >= num_import_tree_row_ptr || import_tree_row_ptr[i] > import_tree_row_ptr[i + 1])) {
                break;
            }

            modules[i]->num_imports = (uint32_t)(import_tree_row_ptr[i + 1] - import_tree_row_ptr[i]);
            modules[i]->allocated_imports_size = modules[i]->num_imports;
            if (modules[i]->num_imports == 0) {
                continue;
            }
            modules[i]->imports = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * modules[i]->num_imports);

            for (uint32_t j = import_tree_row_ptr[i], x = 0; j < import_tree_row_ptr[i + 1]; ++j, x++) {
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
                TVM_RT_WASM_TrieInsertAll((*lib_module)->env_funcs_map, modules[i]->module_funcs_map);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(modules);
        TVM_RT_WASM_HeapMemoryFree(import_tree_row_ptr);
        TVM_RT_WASM_HeapMemoryFree(import_tree_child_indices);
    }
    return 0;
}

/*! \brief the release function for system_lib_module */
static int TVM_RT_WASM_SysLibModuleReleaseFunc(Module *self) {
    MODULE_BASE_MEMBER_FREE(self);
    TVM_RT_WASM_HeapMemoryFree(self->packed_functions);
    TVM_RT_WASM_HeapMemoryFree(self);
    return 0;
}

static void visit_symbol_change_to_func(char c, void **data_ptr, void *source_handle) {
    static int now_functions = 0;
    PackedFunction *pf = (PackedFunction *)source_handle;
    if (*data_ptr != NULL) {
        pf[now_functions].exec = *data_ptr;
        *data_ptr = pf + now_functions;
        ++now_functions;
    }
}

/*!
 * \brief create a system library module (this will be a single instance)
 * @param libraryModule the out handle
 * @return 0 if successful
 */
static int TVM_RT_WASM_SystemLibraryModuleCreate(Module **libraryModule) {
    if (likely(sys_lib_module != NULL)) { // if the instance exists, return
        *libraryModule = sys_lib_module;
        return 0;
    }
    if (unlikely(system_lib_symbol == NULL)) {
        SET_ERROR_RETURN(-1, "no symbol in system library!");
    }

    *libraryModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module));
    memset(*libraryModule, 0, sizeof(Module));

    (*libraryModule)->Release = TVM_RT_WASM_SysLibModuleReleaseFunc;
    (*libraryModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    (*libraryModule)->module_funcs_map = system_lib_symbol;
    TVM_RT_WASM_TrieCreate(&(*libraryModule)->env_funcs_map);

    // dev_blob
    const char *blob = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery((*libraryModule)->module_funcs_map, (const uint8_t *)TVM_DEV_MODULE_BLOB,
                                              (void **)&blob)) {
        int status = TVM_RT_WASM_ModuleLoadBinaryBlob(blob, libraryModule);
        if (unlikely(status)) {
            return status;
        }
    }

    // module context
    void **module_context = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery((*libraryModule)->module_funcs_map, (const uint8_t *)TVM_MODULE_CTX,
                                              (void **)&module_context)) {
        *module_context = *libraryModule;
    } else {
        SET_ERROR_RETURN(-1, "Cannot find module context symbol `%s` from syslib\n", TVM_MODULE_CTX);
    }

    /** init packed function */
    (*libraryModule)->packed_functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_sys_lib_symbol);
    memset((*libraryModule)->packed_functions, 0, sizeof(PackedFunction) * num_sys_lib_symbol);
    TVM_RT_WASM_TrieVisit(system_lib_symbol, visit_symbol_change_to_func, (*libraryModule)->packed_functions);
    system_lib_symbol = NULL; // manage this Trie*

    sys_lib_module = *libraryModule;
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
int TVM_RT_WASM_ModuleFactory(const char *type, const char *resource, int resource_type, Module **out) {
    if (!memcmp(type, MODULE_SYSTEM_LIB, strlen(MODULE_SYSTEM_LIB))) {
        return TVM_RT_WASM_SystemLibraryModuleCreate(out);
    }
    if (!memcmp(type, "so", 2) || !memcmp(type, "dll", 3) || !memcmp(type, "dylib", 5)) {
        if (unlikely(resource_type != MODULE_FACTORY_RESOURCE_FILE)) {
            SET_ERROR_RETURN(-1, "the dso library can only be load from file");
        }
        return TVM_RT_WASM_DSOLibraryModuleCreate(resource, out);
    }
    if (!memcmp(type, "cuda", 4)) {
        return TVM_RT_WASM_CUDAModuleCreate(resource, resource_type, (CUDAModule **)out);
    }
    if (!memcmp(type, "webgpu", 6)) {
        return TVM_RT_WASM_WebGPUModuleCreate(resource, resource_type, (WebGPUModule **)out);
    }
    SET_ERROR_RETURN(-1, "unsupported module type %s\n", type);
}
