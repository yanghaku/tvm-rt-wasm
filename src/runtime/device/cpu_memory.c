/*!
 * \file src/runtime/device/cpu_memory.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/device/cpu_memory.h>

char heap_memory[HEAP_MEMORY_STATIC_SIZE];
uintptr_t heap_now_ptr = heap_memory;

char stack_memory[STACK_MEMORY_STATIC_SIZE];
uintptr_t stack_now_ptr = stack_memory;
uintptr_t stack_alloc_history[STACK_ALLOC_HISTORY_MAX_SIZE];
size_t history_size = 0;
