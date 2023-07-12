/**
 * @file module/module.h
 * @brief Define the module base interface and member.
 */

#ifndef TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_
#define TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_backend_api.h>
#include <utils/trie.h>

typedef struct Module Module;

typedef struct {
    /** @brief The function pointer to execute. */
    TVMBackendPackedCFunc exec;
} PackedFunction;

/** @brief the base interface in module */
#define MODULE_BASE_INTERFACE                                                                      \
    /**                                                                                            \
     * @brief Release the resource for this module.                                                \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*Release)(Module * self);                                                                 \
    /**                                                                                            \
     * @brief Find function from module.                                                           \
     * @param mod The module handle.                                                               \
     * @param func_name The name of the function.                                                  \
     * @param query_imports Whether to query imported modules.                                     \
     * @param out The pointer to save result packed function.                                      \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*GetFunction)(Module * mod, const char *func_name, int query_imports,                     \
                       PackedFunction **out);

/** @brief The base member in module. */
#define MODULE_BASE_MEMBER                                                                         \
    /** @brief the base interfaces. */                                                             \
    MODULE_BASE_INTERFACE                                                                          \
    /** @brief the depend modules array. */                                                        \
    Module **imports;                                                                              \
    /** @brief the cached map <string, PackedFunction*>.                                           \
     *  For "GetFuncFromEnv", save imports + global function.                                      \
     */                                                                                            \
    Trie *env_funcs_map;                                                                           \
    /** @brief the module functions, map <string, PackedFunction*>. */                             \
    Trie *module_funcs_map;                                                                        \
    /** @brief the number of imports. */                                                           \
    size_t num_imports;

/** @brief The base Module. */
struct Module {
    MODULE_BASE_MEMBER
};

/** @brief symbols */
#define TVM_MODULE_CTX "__tvm_module_ctx"
#define TVM_DEV_MODULE_BLOB "__tvm_dev_mblob"
#define TVM_SET_DEVICE_FUNCTION "__tvm_set_device"
#define TVM_MODULE_MAIN "__tvm_main__"
#define TVM_GET_METADATA_FUNC get_c_metadata
#define TVM_GET_METADATA_FUNC_NAME TOSTRING(TVM_GET_METADATA_FUNC)

/**
 * @brief Create a system library module. (It will be a single instance).
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_SystemLibraryModuleCreate(Module **out);

/**
 * @brief Create a library module from the dynamic shared library.
 * @param filename The filename.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_SharedLibraryModuleCreate(const char *filename, Module **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_MODULE_MODULE_H_INCLUDE_
