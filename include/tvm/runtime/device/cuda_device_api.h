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

    int num_device;

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
