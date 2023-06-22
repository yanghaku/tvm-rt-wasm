/*!
 * \file module/dso_module.h
 * \brief Define the dynamic shared library module create function.
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_DSO_MODULE_H
#define TVM_RT_WASM_DSO_MODULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <module/module.h>

#if !USE_WASI_SDK // WASI_SDK no dlopen

#ifdef _MSC_VER
// todo: open shared library in windows
#elif __has_include(<dlfcn.h>)

#include <dlfcn.h>

typedef void *TVM_RT_LIB_NATIVE_HANDLE;
#define TVM_RT_HAS_DSO_LIB 1
#define TVM_RT_WASM_OPEN_LIB dlopen
#define TVM_RT_WASM_CLOSE_LIB dlclose
#define TVM_RT_WASM_FIND_LIB_SYMBOL dlsym

#endif // _MSC_VER

#endif // !USE_WASI_SDK

/*!
 * \brief Create a library module from the dynamic shared library
 * @param filename the filename
 * @param libraryModule the out handle
 * @return 0 if successful
 */
int TVM_RT_WASM_DSOLibraryModuleCreate(const char *filename, Module **dsoModule);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_DSO_MODULE_H
