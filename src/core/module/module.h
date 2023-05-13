/*!
 * \file module/module.h
 * \brief define the module base struct and interface
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_MODULE_H
#define TVM_RT_WASM_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_backend_api.h>
#include <utils/trie.h>

typedef struct Module Module;

typedef struct PackedFunction {
    /*! \brief this function in which module */
    Module *module;
    /*! \brief the function pointer to execute */
    TVMBackendPackedCFunc exec;
    /*! \brief other information, such as function index in module */
    uint64_t reserved;
} PackedFunction;

/*!---------------------------------for the Definition of Module struct-----------------------------------------------*/

/*! \brief the base interface in module */
#define MODULE_BASE_INTERFACE                                                                                          \
    /*!                                                                                                                \
     * \brief Release the resource for this module                                                                     \
     * \return 0 if successful                                                                                         \
     */                                                                                                                \
    int (*Release)(Module * self);

/*! \brief the base member in module */
#define MODULE_BASE_MEMBER                                                                                             \
    /*! \brief the allocated size for imports array */                                                                 \
    uint32_t allocated_imports_size;                                                                                   \
    /*! \brief the number of imports */                                                                                \
    uint32_t num_imports;                                                                                              \
    /*! \brief the depend modules array */                                                                             \
    Module **imports;                                                                                                  \
    /*! \brief the cached map <string, PackedFunction*>, for "GetFuncFromEnv", imports + global function */            \
    Trie *env_funcs_map;                                                                                               \
    /*! \brief the module functions, map <string, PackedFunction*> */                                                  \
    Trie *module_funcs_map;                                                                                            \
    /*! \brief the packed function storage */                                                                          \
    PackedFunction *packed_functions;                                                                                  \
    /*! \brief the base interfaces */                                                                                  \
    MODULE_BASE_INTERFACE

/*! \brief the base class Module */
struct Module {
    MODULE_BASE_MEMBER
};

/*! \brief the public functions for every module */

/*!
 * \brief create a module instance for given type
 * @param type the module type or file format
 * @param resource filename or binary source
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param out the pointer to receive created instance
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
#define MODULE_FACTORY_RESOURCE_BINARY 0
#define MODULE_FACTORY_RESOURCE_FILE 1
int TVM_RT_WASM_ModuleFactory(const char *type, const char *resource, int resource_type, Module **out);

#define MODULE_SYSTEM_LIB "SystemLibrary"

/*! \brief symbols */
#define TVM_MODULE_CTX "__tvm_module_ctx"
#define TVM_DEV_MODULE_BLOB "__tvm_dev_mblob"
#define TVM_SET_DEVICE_FUNCTION "__tvm_set_device"

#define MODULE_BASE_MEMBER_FREE(module)                                                                                \
    do {                                                                                                               \
        if (module->imports) {                                                                                         \
            for (uint32_t i = 0; i < module->num_imports; ++i) {                                                       \
                module->imports[i]->Release(module->imports[i]);                                                       \
            }                                                                                                          \
            TVM_RT_WASM_HeapMemoryFree(module->imports);                                                               \
        }                                                                                                              \
        if (module->module_funcs_map) {                                                                                \
            TVM_RT_WASM_TrieRelease(module->module_funcs_map);                                                         \
        }                                                                                                              \
        if (module->env_funcs_map) {                                                                                   \
            TVM_RT_WASM_TrieRelease(module->env_funcs_map);                                                            \
        }                                                                                                              \
    } while (0)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_MODULE_H
