/*!
 * \file device/device_api.c
 * \brief implement for device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/device_api.h>
#include <utils/common.h>

#define DEVICE_API_CREATE_IF_NO_SUPPORT(dev)                                                       \
    _Pragma(TOSTRING(weak TVM_RT_WASM_##dev##DeviceAPICreate));                                    \
    int TVM_RT_WASM_##dev##DeviceAPICreate(DeviceAPI **out) {                                      \
        *out = NULL;                                                                               \
        TVM_RT_##dev##_NOT_LINK();                                                                 \
        return -1;                                                                                 \
    }

DEVICE_API_CREATE_IF_NO_SUPPORT(CUDA)
DEVICE_API_CREATE_IF_NO_SUPPORT(WebGPU)

/*!
 * \brief the device api for every device type
 */
#define DEVICE_TYPE_SIZE 16
static DeviceAPI *g_device_api_instance[DEVICE_TYPE_SIZE] = {NULL};

/*!
 * \brief get the device api instance for the given device type
 * @param device_type device type
 * @param out_device_api the pointer to receive the point
 * @return 0 if successful
 */
int TVM_RT_WASM_DeviceAPIGet(DLDeviceType device_type, DeviceAPI **out_device_api) {
    int status = 0;
    if (unlikely(g_device_api_instance[device_type] == NULL)) { // need create
        switch (device_type) {
        case kDLCPU:
            TVM_RT_SET_ERROR_RETURN(-1, "CPU device is not used.");
        case kDLCUDA:
        case kDLCUDAHost:
            status = TVM_RT_WASM_CUDADeviceAPICreate(&g_device_api_instance[device_type]);
            if (unlikely(status)) {
                g_device_api_instance[device_type] = NULL;
                return status;
            }
            break;
        case kDLWebGPU:
            status = TVM_RT_WASM_WebGPUDeviceAPICreate(&g_device_api_instance[device_type]);
            if (unlikely(status)) {
                g_device_api_instance[device_type] = NULL;
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
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported device %d", device_type);
        }
    }

    *out_device_api = g_device_api_instance[device_type];
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
