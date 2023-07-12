/**
 * @file module/shared_library.c
 * @brief Implementation for shared library module.
 */

#if USE_WASI_SDK // WASI_SDK no dlopen
#define TVM_RT_HAS_SHARED_LIB 0
#else            // NO WASI_SDK
#ifdef _MSC_VER
// todo: open shared library in windows
#elif __has_include(<dlfcn.h>)

#include <dlfcn.h>

typedef void *SharedLibNativeHandle;
#define TVM_RT_HAS_SHARED_LIB 1
#define TVM_RT_WASM_OPEN_LIB dlopen
#define TVM_RT_WASM_CLOSE_LIB dlclose
#define TVM_RT_WASM_FIND_LIB_SYMBOL dlsym

#endif // _MSC_VER
#endif // USE_WASI_SDK

#if TVM_RT_HAS_SHARED_LIB

#include <device/cpu_memory.h>
#include <module/module_impl.h>

/** @brief SharedLibraryModule, derive from Module. */
typedef struct SharedLibraryModule {
    MODULE_BASE_MEMBER

    SharedLibNativeHandle lib_handle;
} SharedLibraryModule;

/** @brief Free all packed functions for shared library module. */
static void TVM_RT_WASM_TrieVisit_SharedLibPackedFuncFree(void **data_ptr, void *_) {
    (void)_;
    void *p = *data_ptr;
    if (p != NULL) {
        TVM_RT_WASM_HeapMemoryFree(p);
    }
}

/** @brief The release function for shared library module. */
static int TVM_RT_WASM_DsoModuleReleaseFunc(Module *mod) {
    TVM_RT_WASM_TrieVisit(mod->module_funcs_map, TVM_RT_WASM_TrieVisit_SharedLibPackedFuncFree,
                          NULL);
    TVM_RT_WASM_CLOSE_LIB(((SharedLibraryModule *)mod)->lib_handle);
    MODULE_BASE_MEMBER_FREE(mod);

    TVM_RT_WASM_HeapMemoryFree(mod);
    return 0;
}

/** @brief Get packed functions from the module. */
static int TVM_RT_WASM_DsoLibraryGetFunction(Module *mod, const char *func_name, int query_imports,
                                             PackedFunction **out) {
    int status =
        TVM_RT_WASM_TrieQuery(mod->module_funcs_map, (const uint8_t *)func_name, (void **)out);
    if (likely(status != TRIE_NOT_FOUND)) {
        return status;
    }

    // create the packed function and insert to module_funcs_map
    SharedLibNativeHandle native_handle = ((SharedLibraryModule *)mod)->lib_handle;
    if (native_handle) {
        TVMBackendPackedCFunc symbol =
            (TVMBackendPackedCFunc)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, func_name);
        if (symbol) {
            PackedFunction *pf =
                (PackedFunction *)TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction));
            pf->exec = symbol;
            TVM_RT_WASM_TrieInsert(mod->module_funcs_map, (const uint8_t *)func_name, (void *)pf);
            *out = pf;
            return 0;
        }
    }

    if (query_imports) {
        status =
            TVM_RT_WASM_TrieQuery(mod->env_funcs_map, (const uint8_t *)func_name, (void **)out);
        if (status && mod->imports) {
            for (size_t i = 0; i < mod->num_imports; ++i) {
                Module *m = mod->imports[i];
                if (m) {
                    status = m->GetFunction(m, func_name, query_imports, out);
                    if (status == 0) {
                        return status;
                    }
                }
            }
        }
    }
    return status;
}

int TVM_RT_WASM_SharedLibraryModuleCreate(const char *filename, Module **out_module) {
    SharedLibNativeHandle native_handle = TVM_RT_WASM_OPEN_LIB(filename, RTLD_LAZY);
    if (unlikely(native_handle == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot load shared library %s", filename);
    }

    *out_module = TVM_RT_WASM_HeapMemoryAlloc(sizeof(SharedLibraryModule));
    memset(*out_module, 0, sizeof(SharedLibraryModule));

    SharedLibraryModule *mod = (SharedLibraryModule *)*out_module;
    mod->Release = TVM_RT_WASM_DsoModuleReleaseFunc;
    mod->GetFunction = TVM_RT_WASM_DsoLibraryGetFunction;
    mod->lib_handle = native_handle;
    TVM_RT_WASM_TrieCreate(&(mod->module_funcs_map));
    TVM_RT_WASM_TrieCreate(&(mod->env_funcs_map));

    // dev_blob
    const char *blob =
        (const char *)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, TVM_DEV_MODULE_BLOB);
    if (blob) {
        int status = TVM_RT_WASM_LibraryModuleLoadBinaryBlob(blob, out_module);
        if (unlikely(status)) {
            TVM_RT_WASM_DsoModuleReleaseFunc(*out_module);
            *out_module = NULL;
            return status;
        }
    }

    // module context
    void **module_context = (void **)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, TVM_MODULE_CTX);
    if (likely(module_context)) {
        *module_context = *out_module;
    }

#define TVM_RT_WASM_INIT_CONTEXT_FUNC(name)                                                        \
    do {                                                                                           \
        void **symbol = (void **)TVM_RT_WASM_FIND_LIB_SYMBOL(native_handle, "__" #name);           \
        if (symbol) {                                                                              \
            *symbol = name;                                                                        \
        }                                                                                          \
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

#include <module/module.h>
#include <stdio.h>
#include <stdlib.h>

int TVM_RT_WASM_DSOLibraryModuleCreate(const char *filename, Module **out) {
    (void)filename;
    (void)out;
    fprintf(stderr, "Cannot load dynamic shared library in this platform!\n");
    exit(-1);
}

#endif // !TVM_RT_HAS_SHARED_LIB
