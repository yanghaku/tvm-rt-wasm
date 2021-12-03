/*!
 * \file runtime/device/cpu_device_api.h
 * \brief cpu device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_CPU_DEVICE_API_H
#define TVM_RT_CPU_DEVICE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/device/device_api.h>

/*! \brief CPUDeviceAPI implement the interface DeviceAPI */
typedef struct CPUDeviceAPI {
    DEVICE_API_INTERFACE
    // no data
} CPUDeviceAPI;

/*!
 * \brief create the cpu_device_api instance
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int CPUDeviceAPICreate(CPUDeviceAPI **out);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CPU_DEVICE_API_H
