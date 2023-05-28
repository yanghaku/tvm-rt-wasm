/*!
 * \file module/dso_module.c
 * \brief implement functions for dso_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <module/dso_module.h>

#if TVM_RT_HAS_DSO_LIB

/*! \brief Free all packed functions for dso module. */
static void visit_dso_funcs_free(char c, void **data_ptr, void *_) {
    if (*data_ptr != NULL) {
        TVM_RT_WASM_HeapMemoryFree(*data_ptr);
    }
}

/*! \brief the release function for dso_lib_module */
static int TVM_RT_WASM_DsoModuleReleaseFunc(Module *self) {
    TVM_RT_WASM_TrieVisit(self->module_funcs_map, visit_dso_funcs_free, NULL);
    TVM_RT_LIB_NATIVE_HANDLE native_handle = (TVM_RT_LIB_NATIVE_HANDLE)self->packed_functions;
    TVM_RT_WASM_CLOSE_LIB(native_handle);
    MODULE_BASE_MEMBER_FREE(self);

    TVM_RT_WASM_HeapMemoryFree(self);
    return 0;
}

static int TVM_RT_WASM_DsoLibraryGetFunction(Module *mod, const char *func_name, int query_imports,
                                             TVMFunctionHandle *out) {
    int status = TVM_RT_WASM_TrieQuery(mod->module_funcs_map, (const uint8_t *)func_name, out);
    if (likely(status != TRIE_NOT_FOUND)) {
        return status;
    }

    // create the packed function and insert to module_funcs_map
    TVM_RT_LIB_NATIVE_HANDLE native_handle = (TVM_RT_LIB_NATIVE_HANDLE)mod->packed_functions;
    if (native_handle) {
        TVMBackendPackedCFunc symbol = (TVMBackendPackedCFunc)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, func_name);
        if (symbol) {
            PackedFunction *pf = (PackedFunction *)TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction));
            pf->module = mod;
            pf->reserved = 0;
            pf->exec = symbol;
            TVM_RT_WASM_TrieInsert(mod->module_funcs_map, func_name, (void *)pf);
            *out = (TVMFunctionHandle)pf;
            return 0;
        }
    }

    if (query_imports) {
        status = TVM_RT_WASM_TrieQuery(mod->env_funcs_map, (const uint8_t *)func_name, out);
    }
    return status;
}

/*!
 * \brief Create a library module from the dynamic shared library.
 * @param filename the filename
 * @param libraryModule the out handle
 * @return 0 if successful
 */
int TVM_RT_WASM_DSOLibraryModuleCreate(const char *filename, Module **dsoModule) {
    TVM_RT_LIB_NATIVE_HANDLE native_handle = TVM_RT_WASM_OPEN_LIB(filename, RTLD_LAZY);
    if (unlikely(native_handle == NULL)) {
        SET_ERROR_RETURN(-1, "Cannot load shared library %s", filename);
    }

    *dsoModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module));
    memset(*dsoModule, 0, sizeof(Module));

    (*dsoModule)->Release = TVM_RT_WASM_DsoModuleReleaseFunc;
    (*dsoModule)->GetFunction = TVM_RT_WASM_DsoLibraryGetFunction;
    // use packed_functions field to save native library handle.
    (*dsoModule)->packed_functions = (PackedFunction *)native_handle;
    TVM_RT_WASM_TrieCreate(&(*dsoModule)->module_funcs_map);
    TVM_RT_WASM_TrieCreate(&(*dsoModule)->env_funcs_map);

    // dev_blob
    const char *blob = (const char *)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, TVM_DEV_MODULE_BLOB);
    if (blob) {
        int status = TVM_RT_WASM_ModuleLoadBinaryBlob(blob, dsoModule);
        if (unlikely(status)) {
            return status;
        }
    }

    // module context
    void **module_context = (void **)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, TVM_MODULE_CTX);
    if (likely(module_context)) {
        *module_context = *dsoModule;
    }

#define TVM_RT_WASM_INIT_CONTEXT_FUNC(name)                                                                            \
    do {                                                                                                               \
        void **symbol = (void **)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, "__" #name);                               \
        if (symbol) {                                                                                                  \
            *symbol = name;                                                                                            \
        }                                                                                                              \
    } while (0)

    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMFuncCall);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMAPISetLastError);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
    TVM_RT_WASM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);

#undef TVM_RT_WASM_INIT_CONTEXT_FUNC
    return 0;
}

#else

#include <stdio.h>
#include <stdlib.h>

int TVM_RT_WASM_DSOLibraryModuleCreate(const char *filename, Module **dsoModule) {
    fprintf(stderr, "Cannot load dynamic shared library in this platform!\n");
    exit(-1);
}

#endif // !TVM_RT_HAS_DSO_LIB
