/*!
 * \file device/cuda_device_api.h
 * \brief cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_CUDA_DEVICE_API_H
#define TVM_RT_WASM_CUDA_DEVICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <device/device_api.h>
#include <utils/cuda_common.h>

/*! \brief CUDADeviceAPI implement the interface DeviceAPI */
typedef struct CUDADeviceAPI {
    DEVICE_API_INTERFACE

    uint32_t num_device;

#if USE_CUDA // USE_CUDA = 1
    /*! \brief the cuda contexts for every devices */
    CUcontext *contexts;
    /*! \brief the now work stream */
    CUstream stream;

#ifndef CUDA_10_ONLY
    /*! \brief the cuda memory pool for workplace **/
    CUmemoryPool mem_pool;
#endif // CUDA_10_ONLY

#endif // USE_CUDA

} CUDADeviceAPI;

/*!
 * \brief create a instance of cuda device api
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int TVM_RT_WASM_CUDADeviceAPICreate(CUDADeviceAPI **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CUDA_DEVICE_API_H
