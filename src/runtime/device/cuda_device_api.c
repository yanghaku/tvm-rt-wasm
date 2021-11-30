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
 * \file src/runtime/device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/device/cuda_device_api.h>

/*!
 * \brief create a instance of cuda device api
 * @param cudaDeviceApi the pointer to receive instance
 * @return 0 if successful
 */
int CUDADeviceAPICreate(CUDADeviceAPI **cudaDeviceApi) {

#if USE_CUDA // USE_CUDA = 1

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(cudaDeviceApi), 0, no_type, (void **)&cudaDeviceApi);

    CUDA_CALL(cudaGetDeviceCount(&(*cudaDeviceApi)->num_device));
    // todo: init CUcontext
    return 0;

#else
    fprintf(stderr, "CUDA library is not supported! you can compile from source and set USE_CUDA option ON\n");
    exit(-1);
#endif
}
