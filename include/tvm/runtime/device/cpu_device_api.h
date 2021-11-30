/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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
 * @param cpuDeviceApi the pointer to receive instance
 * @return 0 if successful
 */
int CPUDeviceAPICreate(CPUDeviceAPI **cpuDeviceApi);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CPU_DEVICE_API_H
