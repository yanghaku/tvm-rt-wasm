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

#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/utils/common.h>

extern char g_memory[];
extern uintptr_t now;

#define alignment_up(a, size) (((a) + (size)-1) & (~((size)-1)))
#define ALIGNMENT_SIZE 8

/*!
 * \brief malloc the heap memory
 * @param bytes the number of bytes to allocate
 * @return the pointer
 */
INLINE void *TVM_RT_WASM_HeapMemoryAlloc(size_t bytes) {
    void *ans = (void *)now;
    now = alignment_up(now + bytes, ALIGNMENT_SIZE);
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
    void *ans = (void *)now;
    now = alignment_up(now + bytes, ALIGNMENT_SIZE);
    return ans;
}

/*!
 * \brief free the temporal memory
 * @param ptr
 */
INLINE void TVM_RT_WASM_WorkplaceMemoryFree(void *ptr) {}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CPU_MEMORY_H
