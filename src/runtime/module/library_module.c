/*!
 * \file src/runtime/module/library_module.c
 * \brief implement functions for library_module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/module/library_module.h>

/*! \brief the symbols for system library */
Trie *system_lib_symbol = NULL;

/*! \brief the system library module is a single instance */
static Module *sys_lib_module = NULL;
