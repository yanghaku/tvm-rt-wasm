/**
 * @file device/cpu_memory.h
 * @brief CPU memory alloc/free interface.
 */

#ifndef TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_
#define TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Up to the multiple of (1<<_bits) */
#define ALIGN_UP(_a, _bits) (((_a) + (1 << (_bits)) - 1) & (~((1 << (_bits)) - 1)))

/** @brief Default data alignment is 64 (1<<6) */
#ifndef DATA_ALIGNMENT_BITS
#define DATA_ALIGNMENT_BITS 6
#endif // DATA_ALIGNMENT_BITS

#include <stdlib.h>

#define TVM_RT_WASM_HeapMemoryAlignedAlloc(bytes)                                                  \
    aligned_alloc((1 << DATA_ALIGNMENT_BITS), ALIGN_UP(bytes, DATA_ALIGNMENT_BITS))
#define TVM_RT_WASM_HeapMemoryAlloc malloc
#define TVM_RT_WASM_HeapMemoryFree free

#define TVM_RT_WASM_WorkplaceMemoryAlloc TVM_RT_WASM_HeapMemoryAlignedAlloc
#define TVM_RT_WASM_WorkplaceMemoryFree free

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_DEVICE_CPU_MEMORY_H_INCLUDE_
