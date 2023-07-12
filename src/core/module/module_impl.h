/**
 * @file module/module_impl.h
 * @brief Private module interface, only for module implementation files.
 */

#ifndef TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_
#define TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <module/module.h>
#include <utils/common.h>

#define MODULE_BASE_MEMBER_FREE(_mod)                                                              \
    do {                                                                                           \
        if ((_mod)->imports) {                                                                     \
            for (uint32_t i = 0; i < (_mod)->num_imports; ++i) {                                   \
                if ((_mod)->imports[i]) {                                                          \
                    (_mod)->imports[i]->Release((_mod)->imports[i]);                               \
                }                                                                                  \
            }                                                                                      \
            TVM_RT_WASM_HeapMemoryFree((_mod)->imports);                                           \
        }                                                                                          \
        if ((_mod)->module_funcs_map) {                                                            \
            TVM_RT_WASM_TrieRelease((_mod)->module_funcs_map);                                     \
        }                                                                                          \
        if ((_mod)->env_funcs_map) {                                                               \
            TVM_RT_WASM_TrieRelease((_mod)->env_funcs_map);                                        \
        }                                                                                          \
    } while (0)

/** @brief Default implementation for module get function. */
int TVM_RT_WASM_DefaultModuleGetFunction(Module *mod, const char *func_name, int query_imports,
                                         PackedFunction **out);

/**
 * @brief Load modules tree from binary blob.
 * @param blob the dev_blob binary.
 * @param lib_module The root library module handle.
 * @return 0 if successful
 * @note It can only be used in library module.
 */
int TVM_RT_WASM_LibraryModuleLoadBinaryBlob(const char *blob, Module **lib_module);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_MODULE_MODULE_IMPL_H_INCLUDE_
