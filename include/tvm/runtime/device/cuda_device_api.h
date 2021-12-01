/*!
 * \file runtime/device/cuda_device_api.h
 * \brief cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_CUDA_DEVICE_API_H
#define TVM_RT_CUDA_DEVICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/device/device_api.h>

#if USE_CUDA // USE_CUDA = 1
#include <tvm/runtime/utils/cuda_common.h>
#endif // USE_CUDA

/*! \brief CUDADeviceAPI implement the interface DeviceAPI */
typedef struct CUDADeviceAPI {
    DEVICE_API_INTERFACE

    uint32_t num_device;

#if USE_CUDA // USE_CUDA = 1
    CUcontext *contexts;
#elif // USE_CUDA

#endif
} CUDADeviceAPI;

/*!
 * \brief create a instance of cuda device api
 * @param cudaDeviceApi the pointer to receive instance
 * @return 0 if successful
 */
int CUDADeviceAPICreate(CUDADeviceAPI **cudaDeviceApi);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CUDA_DEVICE_API_H
