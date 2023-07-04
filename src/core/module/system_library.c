/*!
 * \file module/system_library.c
 * \brief implementation for system library module
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <module/module.h>
#include <tvm/runtime/c_backend_api.h>

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

/*! \brief the release function for system_lib_module */
static int TVM_RT_WASM_SysLibModuleReleaseFunc(Module *self) {
    MODULE_BASE_MEMBER_FREE(self);
    if (self->packed_functions) {
        TVM_RT_WASM_HeapMemoryFree(self->packed_functions);
    }
    TVM_RT_WASM_HeapMemoryFree(self);
    return 0;
}

static void visit_symbol_change_to_func(void **data_ptr, void *source_handle) {
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
int TVM_RT_WASM_SystemLibraryModuleCreate(Module **libraryModule) {
    if (likely(sys_lib_module != NULL)) { // if the instance exists, return
        *libraryModule = sys_lib_module;
        return 0;
    }
    if (unlikely(system_lib_symbol == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "No symbol in system library!");
    }

    int status;
    *libraryModule = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module));
    memset(*libraryModule, 0, sizeof(Module));

    (*libraryModule)->Release = TVM_RT_WASM_SysLibModuleReleaseFunc;
    (*libraryModule)->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;
    TVM_RT_WASM_TrieCreate(&(*libraryModule)->env_funcs_map);

    // dev_blob
    const char *blob = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(system_lib_symbol,
                                              (const uint8_t *)TVM_DEV_MODULE_BLOB,
                                              (void **)&blob)) {
        status = TVM_RT_WASM_ModuleLoadBinaryBlob(blob, libraryModule);
        if (unlikely(status)) {
            goto sys_lib_fail;
        }
    }

    // module context
    void **module_context = NULL;
    if (TRIE_SUCCESS == TVM_RT_WASM_TrieQuery(system_lib_symbol, (const uint8_t *)TVM_MODULE_CTX,
                                              (void **)&module_context)) {
        *module_context = *libraryModule;
    } else {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(sys_lib_fail,
                                  "Cannot find module context symbol `%s` from system library",
                                  TVM_MODULE_CTX);
    }

    /** init packed function */
    (*libraryModule)->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(PackedFunction) * num_sys_lib_symbol);
    memset((*libraryModule)->packed_functions, 0, sizeof(PackedFunction) * num_sys_lib_symbol);
    TVM_RT_WASM_TrieVisit(system_lib_symbol, visit_symbol_change_to_func,
                          (*libraryModule)->packed_functions);
    (*libraryModule)->module_funcs_map = system_lib_symbol;
    system_lib_symbol = NULL; // manage this Trie*

    sys_lib_module = *libraryModule;
    return 0;

sys_lib_fail:
    TVM_RT_WASM_SysLibModuleReleaseFunc(*libraryModule);
    return status;
}
