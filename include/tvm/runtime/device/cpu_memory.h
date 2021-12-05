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

#include <stdlib.h>
#include <tvm/runtime/utils/common.h>

/*!
 * \brief malloc the heap memory
 * @param bytes the number of bytes to allocate
 * @return the pointer
 */
INLINE void *TVM_RT_WASM_HeapMemoryAlloc(size_t bytes) { return malloc(bytes); }

INLINE void TVM_RT_WASM_HeapMemoryFree(void *ptr) { free(ptr); }

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CPU_MEMORY_H
