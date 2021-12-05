/*!
 * \file runtime/module/module.h
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
#include <tvm/runtime/utils/trie.h>

/*! \brief for TVMFunctionHandle encode and decoce */
/**
 *  In WebAssembly, every function pointer will be stored in indirect call table.
 *  We assume that the indirect call table size will not larger than 3/4 of pointer size
 *
 *  in wasm32: TVMFunctionHandle = (void*) = 32bit,  32bit = resource(high 16bit) + TVMBackendPackedCFunc (low 16bit)
 *  in wasm64: TVMFunctionHandle = (void*) = 64bit,  64bit = resource(high 32bit) + TVMBackendPackedCFunc (low 32bit)
 *
 */

#ifndef __SIZEOF_POINTER__
#if UINTPTR_MAX == UUINT32_MAX
#define __SIZEOF_POINTER__ 4
#elif UINTPTR_MAX == UINT64_MAX
#define __SIZEOF_POINTER__ 8
#else
#error "can not get the size of pointer"
#endif
#endif

#define ENCODE_SHIFT_BITS ((__SIZEOF_POINTER__) << 2)
#define ENCODE_AND_VALUES ((((uintptr_t)1) << (ENCODE_SHIFT_BITS)) - 1)
#define TVM_FUNCTION_HANDLE_ENCODE(backend_func, resource)                                                             \
    ((TVMFunctionHandle)(((uintptr_t)(backend_func)) | (((uintptr_t)(resource)) << ENCODE_SHIFT_BITS)))
#define TVM_FUNCTION_HANDLE_DECODE_EXEC(func_handle)                                                                   \
    ((TVMBackendPackedCFunc)(((uintptr_t)(func_handle)) & (ENCODE_AND_VALUES)))
#define TVM_FUNCTION_HANDLE_DECODE_RESOURCE(func_handle) (((uintptr_t)(func_handle)) >> ENCODE_SHIFT_BITS)

/*!---------------------------------for the Definition of Module struct-----------------------------------------------*/

typedef struct Module Module;

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
    /*! \brief the cached map <string, TVMFunctionHandle>, for "GetFuncFromEnv", imports + global function */          \
    Trie *env_funcs_map;                                                                                               \
    /*! \brief the module functions, map <string, TVMFunctionHandle> */                                                \
    Trie *module_funcs_map;                                                                                            \
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

/*! \brief the symbols for system library */
extern Trie *system_lib_symbol;

#define MODULE_SYSTEM_LIB "SystemLibrary"

/*! \brief symbols */
#define TVM_MODULE_CTX "__tvm_module_ctx"
#define TVM_DEV_MODULE_BLOB "__tvm_dev_mblob"
#define TVM_SET_DEVICE_FUNCTION "__tvm_set_device"

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_MODULE_H
