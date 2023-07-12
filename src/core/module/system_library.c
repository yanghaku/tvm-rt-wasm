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
 * @note The sys_lib_module instance will manage this trie after the init.
 */
static Trie *system_lib_symbol = NULL;
static size_t num_sys_lib_symbol = 0;

/** @brief The system library module is a single instance */
static SystemLibraryModule *sys_lib_module = NULL;

/** @brief Destroy the single instance when exit. */
static TVM_ATTRIBUTE_UNUSED __attribute__((destructor)) void TVM_RT_WASM_Destructor_SysLib() {
    if (sys_lib_module) {
        MODULE_BASE_MEMBER_FREE(sys_lib_module);
        if (sys_lib_module->packed_functions) {
            TVM_RT_WASM_HeapMemoryFree(sys_lib_module->packed_functions);
        }
        TVM_RT_WASM_HeapMemoryFree(sys_lib_module);
        sys_lib_module = NULL;
    }

    if (system_lib_symbol) {
        TVM_RT_WASM_TrieRelease(system_lib_symbol);
    }
}

/**
 * @brief Backend function to register system-wide library symbol.
 * @sa tvm/runtime/c_backend_api.h
 */
TVM_DLL TVM_ATTRIBUTE_UNUSED int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
    if (unlikely(system_lib_symbol == NULL)) {
        TVM_RT_WASM_TrieCreate(&system_lib_symbol);
    }
    ++num_sys_lib_symbol;
    return TVM_RT_WASM_TrieInsert(system_lib_symbol, (const uint8_t *)name, ptr);
}

/** @brief The release function for system library module. */
static int TVM_RT_WASM_SysLibModuleReleaseFunc(Module *mod) {
    // do nothing
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
    if (likely(sys_lib_module != NULL)) { // if the instance exists, return
        *out_module = (Module *)sys_lib_module;
        return 0;
    }
    if (unlikely(system_lib_symbol == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "No symbol in system library!");
    }

    int status;
    *out_module = TVM_RT_WASM_HeapMemoryAlloc(sizeof(SystemLibraryModule));
    memset(*out_module, 0, sizeof(SystemLibraryModule));

    (*out_module)->Release = TVM_RT_WASM_SysLibModuleReleaseFunc;
    (*out_module)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&((*out_module)->env_funcs_map));

    // dev_blob
    const char *blob = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(system_lib_symbol,
                                              (const uint8_t *)TVM_DEV_MODULE_BLOB,
                                              (void **)&blob)) {
        status = TVM_RT_WASM_LibraryModuleLoadBinaryBlob(blob, out_module);
        if (unlikely(status)) {
            goto sys_lib_fail;
        }
    }

    // module context
    void **module_context = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(system_lib_symbol, (const uint8_t *)TVM_MODULE_CTX,
                                              (void **)&module_context)) {
        *module_context = *out_module;
    } else {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(sys_lib_fail,
                                  "Cannot find module context symbol `%s` from system library",
                                  TVM_MODULE_CTX);
    }

    /** init packed function */
    SystemLibraryModule *mod = (SystemLibraryModule *)(*out_module);

    mod->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_sys_lib_symbol);
    memset(mod->packed_functions, 0, sizeof(PackedFunction) * num_sys_lib_symbol);
    TVM_RT_WASM_TrieVisit(system_lib_symbol, TVM_RT_WASM_TrieVisit_ChangeSymbolToPackedFunc,
                          mod->packed_functions);

    // manage this Trie*
    mod->module_funcs_map = system_lib_symbol;
    system_lib_symbol = NULL;
    num_sys_lib_symbol = 0;

    sys_lib_module = mod;
    return 0;

sys_lib_fail:
    TVM_RT_WASM_SysLibModuleReleaseFunc(*out_module);
    *out_module = NULL;
    return status;
}
