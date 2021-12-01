/*!
 * \file runtime/module/library_module.h
 * \brief define the library module derived from Module, it contain system lib and dynamic lib
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_LIBRARY_MODULE_H
#define TVM_RT_LIBRARY_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/module/module.h>

/*! \brief define the library module derived from module */
typedef struct LibraryModule {
    MODULE_BASE_MEMBER
} LibraryModule;

/*! \brief the symbols for system library */
extern Trie *system_lib_symbol;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_LIBRARY_MODULE_H
