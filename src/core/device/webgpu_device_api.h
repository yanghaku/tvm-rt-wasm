/*!
 * \file device/webgpu_device_api.h
 * \brief webgpu device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_WEBGPU_DEVICE_API_H
#define TVM_RT_WASM_WEBGPU_DEVICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <device/device_api.h>
#include <utils/webgpu_common.h>

/*! \brief WebGPUDeviceAPI implement the interface DeviceAPI */
typedef struct WebGPUDeviceAPI {
    DEVICE_API_INTERFACE

#if USE_WEBGPU // USE_WEBGPU == 1

    WGPU_Device *devices;

    int num_device;

#endif // USE_WEBGPU

} WebGPUDeviceAPI;

/*!
 * \brief create a instance of WebGPU device api
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int TVM_RT_WASM_WebGPUDeviceAPICreate(WebGPUDeviceAPI **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_WEBGPU_DEVICE_API_H
