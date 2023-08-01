/**
 * @file module/system_library.c
 * @brief Implementation for system library module.
 */

#include <module/module_impl.h>
#include <tvm/runtime/c_runtime_api.h>

/** @brief SystemLibraryModule, derive from Module. */
typedef struct SystemLibraryModule {
    MODULE_BASE_MEMBER

    PackedFunction *packed_functions;
} SystemLibraryModule;

/**
 * @brief The symbols for system library.
 * @note The system library module instance will manage this trie after created.
 */
static Trie *sys_lib_symbols = NULL;

/** @brief The system library root module handle. (single instance). */
static Module *sys_lib_root_mod_handle = NULL;

/**
 * if sys_lib_symbols is not NULL, the num_sys_lib_symbol is valid.
 * if sys_lib_root_mod_handle is not NULL, the sys_lib_root_mod_release_func is valid.
 */
static union {
    /** @brief The number of system library symbols. */
    size_t num_sys_lib_symbol;
    /** @brief The system library root module release function. */
    int (*sys_lib_root_mod_release_func)(Module *self);
} sys_lib_status = {.num_sys_lib_symbol = 0};

/** @brief Destroy the single instance when exit. */
static TVM_ATTRIBUTE_UNUSED __attribute__((destructor)) void TVM_RT_WASM_Destructor_SysLib() {
    if (sys_lib_root_mod_handle) {
        sys_lib_status.sys_lib_root_mod_release_func(sys_lib_root_mod_handle);
    } else if (sys_lib_symbols) {
        TVM_RT_WASM_TrieRelease(sys_lib_symbols);
    }
}

/**
 * @brief Backend function to register system-wide library symbol.
 * @sa tvm/runtime/c_backend_api.h
 */
TVM_DLL TVM_ATTRIBUTE_UNUSED int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
    if (unlikely(sys_lib_symbols == NULL)) {
        if (unlikely(sys_lib_root_mod_handle != NULL)) {
            TVM_RT_SET_ERROR_RETURN(
                -1, "Cannot register symbol! The system library module has been created!");
        }
        TVM_RT_WASM_TrieCreate(&sys_lib_symbols);
    }
    ++sys_lib_status.num_sys_lib_symbol;
    return TVM_RT_WASM_TrieInsert(sys_lib_symbols, (const uint8_t *)name, ptr);
}

/** @brief The release function for system library module. */
static int TVM_RT_WASM_SysLibModuleReleaseFunc(Module *mod) {
    SystemLibraryModule *sys_lib_module = (SystemLibraryModule *)mod;
    MODULE_BASE_MEMBER_FREE(sys_lib_module);
    if (sys_lib_module->packed_functions) {
        TVM_RT_WASM_HeapMemoryFree(sys_lib_module->packed_functions);
    }
    TVM_RT_WASM_HeapMemoryFree(sys_lib_module);
    return 0;
}

/** @brief Do nothing function, replace the system library root module release function.
 * The system library root module can only be released when exit.
 */
static int TVM_RT_WASM_ReleaseDoNothing(Module *mod) {
    (void)mod;
    return 0;
}

/** @brief Visit function, change the symbol to packed function. */
static void TVM_RT_WASM_TrieVisit_ChangeSymbolToPackedFunc(void **data_ptr, void *source_handle) {
    static int now_functions = 0;
    PackedFunction *pf = (PackedFunction *)source_handle;
    void *data = *data_ptr;
    if (data != NULL) {
        pf[now_functions].exec = data;
        *data_ptr = pf + now_functions;
        ++now_functions;
    }
}

int TVM_RT_WASM_SystemLibraryModuleCreate(Module **out_module) {
    if (likely(sys_lib_root_mod_handle != NULL)) { // if the instance exists, return
        *out_module = sys_lib_root_mod_handle;
        return 0;
    }
    if (unlikely(sys_lib_symbols == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "No symbol in system library!");
    }

    int status;
    *out_module = TVM_RT_WASM_HeapMemoryAlloc(sizeof(SystemLibraryModule));
    memset(*out_module, 0, sizeof(SystemLibraryModule));

    (*out_module)->Release = TVM_RT_WASM_SysLibModuleReleaseFunc;
    (*out_module)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*out_module)->env_funcs_map));

    // save this system library module handle before load dev_blob.
    SystemLibraryModule *sys_lib_module = (SystemLibraryModule *)(*out_module);

    // dev_blob
    const char *blob = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(sys_lib_symbols, (const uint8_t *)TVM_DEV_MODULE_BLOB,
                                              (void **)&blob)) {
        status = TVM_RT_WASM_LibraryModuleLoadBinaryBlob(blob, out_module);
        if (unlikely(status)) {
            goto sys_lib_fail;
        }
    }

    // module context
    void **module_context = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(sys_lib_symbols, (const uint8_t *)TVM_MODULE_CTX,
                                              (void **)&module_context)) {
        *module_context = *out_module;
    } else {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(sys_lib_fail,
                                  "Cannot find module context symbol `%s` from system library",
                                  TVM_MODULE_CTX);
    }

    // init packed functions
    sys_lib_module->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * sys_lib_status.num_sys_lib_symbol);
    memset(sys_lib_module->packed_functions, 0,
           sizeof(PackedFunction) * sys_lib_status.num_sys_lib_symbol);
    TVM_RT_WASM_TrieVisit(sys_lib_symbols, TVM_RT_WASM_TrieVisit_ChangeSymbolToPackedFunc,
                          sys_lib_module->packed_functions);

    // manage this Trie*
    sys_lib_module->module_funcs_map = sys_lib_symbols;
    sys_lib_symbols = NULL;

    sys_lib_root_mod_handle = *out_module;
    // save the release function.
    sys_lib_status.sys_lib_root_mod_release_func = sys_lib_root_mod_handle->Release;
    sys_lib_root_mod_handle->Release = TVM_RT_WASM_ReleaseDoNothing;
    return 0;

sys_lib_fail:
    TVM_RT_WASM_SysLibModuleReleaseFunc(*out_module);
    *out_module = NULL;
    return status;
}
