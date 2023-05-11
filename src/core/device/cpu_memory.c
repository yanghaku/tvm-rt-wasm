/*!
 * \file device/cpu_memory.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#ifdef CPU_STATIC_MEMORY
#include <device/cpu_memory.h>

char heap_memory[HEAP_MEMORY_STATIC_SIZE];
uintptr_t heap_now_ptr = (uintptr_t)heap_memory;

char stack_memory[STACK_MEMORY_STATIC_SIZE];
uintptr_t stack_now_ptr = (uintptr_t)stack_memory;
uintptr_t stack_alloc_history[STACK_ALLOC_HISTORY_MAX_SIZE];
size_t history_size = 0;

#endif // CPU_STATIC_MEMORY
