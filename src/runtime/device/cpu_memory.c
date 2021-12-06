/*!
 * \file src/runtime/device/cpu_memory.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/device/cpu_memory.h>

char g_memory[20000000];
uintptr_t now = (uintptr_t)g_memory;
