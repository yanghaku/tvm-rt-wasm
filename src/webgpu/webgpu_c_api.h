/*!
 * \file webgpu/webgpu_c_api.h
 * \brief webgpu sync api wrapper for native and js.
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_WEBGPU_WEBGPU_C_APU_H
#define TVM_RT_WASM_WEBGPU_WEBGPU_C_APU_H

#if USE_WEBGPU // USE_WEBGPU = 1

#include <stdio.h>

// the error string can be got using `TVMGetLastError`
#define WGPU_CALL(x)                                                                                                   \
    do {                                                                                                               \
        int result = (x);                                                                                              \
        if (unlikely(result)) {                                                                                        \
            return -1;                                                                                                 \
        }                                                                                                              \
    } while (0)

// the error string can be got using `TVMGetLastError`
#define WGPU_CALL_NULL(x)                                                                                              \
    do {                                                                                                               \
        int result = (x);                                                                                              \
        if (unlikely(result)) {                                                                                        \
            return NULL;                                                                                               \
        }                                                                                                              \
    } while (0)

typedef struct WGPU_Device_st *WGPU_Device;

typedef struct WGPU_Memory_st *WGPU_Memory;

typedef struct WGPU_Function_st *WGPU_Function;

/*!
 * \brief Get the device.
 * \param device_ptr The pointer to receive device.
 * \return 0 if success.
 */
int WGPU_DeviceGet(WGPU_Device *device_ptr);

/*!
 * \brief Free the device instance.
 * \param device The device instance.
 * \return 0 if success.
 */
int WGPU_DeviceFree(WGPU_Device device);

/*!
 * \brief Alloc device memory.
 * \param device The device where memory in.
 * \param memory_ptr The pointer to receive allocated device memory.
 * \param nbytes The number of bytes to alloc.
 * \return 0 if success.
 */
int WGPU_MemoryAlloc(WGPU_Device device, WGPU_Memory *memory_ptr, size_t nbytes);

/*!
 * \brief Free the device memory.
 * \param memory The device memory to free.
 * \return 0 if success.
 */
int WGPU_MemoryFree(WGPU_Memory memory);

/*!
 * \brief Copy memory from host to device.
 * \param dst device memory.
 * \param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * \param src source host memory.
 * \param src_byte_offset offset in bytes to the beginning pointer to src data.
 * \param nbytes the number of bytes to copy.
 * \return 0 if success.
 */
int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, void *src, size_t src_byte_offset, size_t nbytes);

/*!
 * \brief Copy memory from device to host.
 * \param dst host memory.
 * \param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * \param src source device memory.
 * \param src_byte_offset offset in bytes to the beginning pointer to src data.
 * \param nbytes the number of bytes to copy.
 * \return 0 if success.
 */
int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes);

/*!
 * \brief Copy memory from device to device.
 * \param dst device memory.
 * \param dst_byte_offset offset in bytes to the beginning pointer to dst data.
 * \param src source device memory.
 * \param src_byte_offset offset in bytes to the beginning pointer to src data.
 * \param nbytes the number of bytes to copy.
 * \return 0 if success.
 */
int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes);

/*!
 * \brief Create device function.
 * \param func_ptr The pointer to receive created function instance.
 * \param source The text device source code.
 * \return 0 if success.
 */
int WGPU_FunctionCreate(WGPU_Function *func_ptr, char *source);

/*!
 * \brief Submit function to gpu to run.
 * \param function The function instance.
 * \param block_dims The dimension of blocks.
 * \param thread_dims The dimension of threads.
 * \param kernel_args The device function arguments.
 * \param num_kernel_args The number of device function arguments.
 * \return 0 if success.
 */
int WGPU_FunctionRun(WGPU_Function function, size_t block_dims[3], size_t thread_dims[3], WGPU_Memory kernel_args[],
                     uint32_t num_kernel_args);

/*!
 * \brief Free the device function.
 * \param function The function to free.
 * \return 0 if success.
 */
int WGPU_FunctionFree(WGPU_Function function);

#endif // USE_WEBGPU

#endif // TVM_RT_WASM_WEBGPU_WEBGPU_C_APU_H
