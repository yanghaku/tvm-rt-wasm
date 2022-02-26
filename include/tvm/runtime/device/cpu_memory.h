/*!
 * \file runtime/device/cpu_memory.h
 * \brief the main memory manager for cpu
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_CPU_MEMORY_H
#define TVM_RT_WASM_CPU_MEMORY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/utils/common.h>

#define HEAP_MEMORY_STATIC_SIZE 602000000
#define STACK_MEMORY_STATIC_SIZE 420000000
#define STACK_ALLOC_HISTORY_MAX_SIZE 1000

extern char heap_memory[];
extern uintptr_t heap_now_ptr;

extern char stack_memory[];
extern uintptr_t stack_now_ptr;
extern uintptr_t stack_alloc_history[];
extern size_t history_size;

#define alignment_up(a, size) (((a) + (size)-1) & (~((size)-1)))
#define ALIGNMENT_SIZE 16

/*!
 * \brief malloc the heap memory
 * @param bytes the number of bytes to allocate
 * @return the pointer
 */
INLINE void *TVM_RT_WASM_HeapMemoryAlloc(size_t bytes) {
    uintptr_t ans = alignment_up(heap_now_ptr, ALIGNMENT_SIZE);
    heap_now_ptr = ans + bytes;
    return ans;
}

/*!
 * \brief free the heap memory
 * @param ptr the pointer
 */
INLINE void TVM_RT_WASM_HeapMemoryFree(void *ptr) {}

/*!
 * \brief malloc the temporal memory
 * @param bytes the number of bytes to allocate
 * @return the pointer
 */
INLINE void *TVM_RT_WASM_WorkplaceMemoryAlloc(size_t bytes) {
    stack_alloc_history[history_size++] = stack_now_ptr;
    uintptr_t ans = alignment_up(stack_now_ptr, ALIGNMENT_SIZE);
    stack_now_ptr = ans + bytes;
    return ans;
}

/*!
 * \brief free the temporal memory
 * @param ptr
 */
INLINE void TVM_RT_WASM_WorkplaceMemoryFree(void *ptr) { stack_now_ptr = stack_alloc_history[--history_size]; }

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CPU_MEMORY_H
