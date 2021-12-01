/*!
 * \file runtime/module/module.h
 * \brief define the module base struct and interface
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_MODULE_H
#define TVM_RT_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/utils/trie.h>

typedef struct Module Module;

/*! \brief the base interface in module */
#define MODULE_BASE_INTERFACE                                                                                          \
    /*!                                                                                                                \
     * \brief Get a func from module.                                                                                  \
     * \param name the name of the function.                                                                           \
     * \param query_imports Whether to query imported modules                                                          \
     * \param func the pointer to receive result function handle                                                       \
     * \return 0 if successful                                                                                         \
     */                                                                                                                \
    int (*GetFunction)(Module * self, const char *name, int query_imports, TVMFunctionHandle *func);                   \
                                                                                                                       \
    /*!                                                                                                                \
     * \brief Get a function from current environment                                                                  \
     *  The environment includes all the imports as well as Global functions.                                          \
     *                                                                                                                 \
     * \param name name of the function.                                                                               \
     * \param func the pointer to receive function handle                                                              \
     * \return 0 if successful                                                                                         \
     */                                                                                                                \
    int (*GetFuncFromEnv)(Module * self, const char *name, TVMFunctionHandle *func);                                   \
                                                                                                                       \
    /*!                                                                                                                \
     * \brief Import another module into this module.                                                                  \
     * \param other The module to be imported.                                                                         \
     *                                                                                                                 \
     * \note Cyclic dependency is not allowed among modules,                                                           \
     *  An error will be thrown when cyclic dependency is detected.                                                    \
     */                                                                                                                \
    int (*Import)(Module * self, Module * other);                                                                      \
                                                                                                                       \
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
    Module *imports;                                                                                                   \
    /*! \brief the cached map <string, TVMBackendFunc>, for "GetFuncFromEnv", imports + global function */             \
    Trie *env_cache_funcs;                                                                                             \
    /*! \brief the base interfaces */                                                                                  \
    MODULE_BASE_INTERFACE

/*! \brief the base class Module */
struct Module {
    MODULE_BASE_MEMBER
};

/*!
 * \brief create a module instance for given type
 * @param type the module type or file format
 * @param resource filename or binary source
 * @param resource_len the len for binary source (if filename, it can be 0)
 * @param out the pointer to receive created instance
 * @return 0 if successful
 */
int ModuleFactory(const char *type, const char *resource, int resource_len, Module **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_MODULE_H
