/*!
 * \file src/runtime/module/module.c
 * \brief implement functions for module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/module/cuda_module.h>
#include <tvm/runtime/module/library_module.h>
#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/common.h>



/*!
 * \brief create a module instance for given type
 * @param type the module type or file format
 * @param resource filename or binary source
 * @param resource_len the len for binary source (if filename, it can be 0)
 * @param out the pointer to receive created instance
 * @return 0 if successful
 */
int ModuleFactory(const char *type, const char *resource, int resource_len, Module **out) {
    return 0;
}
