/*!
 * \file src/runtime/device/device_api.c
 * \brief implement for device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device/cuda_device_api.h>

/*!
 * \brief the device api for every device type
 */
#define DEVICE_TYPE_SIZE 16
static DeviceAPI *g_device_api_instance[DEVICE_TYPE_SIZE] = {NULL};

/*!
 * \brief get the device api instance for the given device type
 * @param deviceType device type
 * @param out_device_api the pointer to receive the point
 * @return 0 if successful
 */
int TVM_RT_WASM_DeviceAPIGet(DLDeviceType deviceType, DeviceAPI **out_device_api) {
    int status = 0;
    if (unlikely(g_device_api_instance[deviceType] == NULL)) { // need create
        switch (deviceType) {
        case kDLCPU:
            SET_ERROR_RETURN(-1, "cpu device is not used in this project");
        case kDLCUDA:
        case kDLCUDAHost:
            status = TVM_RT_WASM_CUDADeviceAPICreate((CUDADeviceAPI **)&g_device_api_instance[deviceType]);
            if (unlikely(status)) {
                return status;
            }
            break;
        case kDLOpenCL:
        case kDLVulkan:
        case kDLMetal:
        case kDLVPI:
        case kDLROCM:
        case kDLROCMHost:
        case kDLExtDev:
        default:
            SET_ERROR_RETURN(-1, "unsupported device!!");
        }
    }

    *out_device_api = g_device_api_instance[deviceType];
    return status;
}

/*!
 * \brief destroy all device api instance
 * @return 0 if successful
 */
void TVM_RT_WASM_DeviceReleaseAll() {
    for (int i = DEVICE_TYPE_SIZE - 1; i >= 0; --i) {
        if (g_device_api_instance[i]) {
            g_device_api_instance[i]->Release(g_device_api_instance[i]);
            g_device_api_instance[i] = NULL;
        }
    }
}
