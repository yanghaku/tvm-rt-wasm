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
 * \file src/runtime/memory.c
 * \brief the implement for std_memory manager
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdlib.h>
#include <tvm/internal/cuda/cuda_common.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>

int memory_alloc(uint32_t num_bytes, DLDevice dev, void **out_ptr) {
    int status;
    switch (dev.device_type) {
    case kDLCUDA:
        status = cudaMalloc(out_ptr, num_bytes);
        if (unlikely(status)) {
            fprintf(stderr, "allocate fail, device=%d, err=%s\n", dev.device_type,
                    cudaGetErrorString(status));
            return status;
        }
        return 0;
    case kDLCUDAHost:
        status = cudaMallocHost(out_ptr, num_bytes);
        if (unlikely(status)) {
            fprintf(stderr, "allocate fail, device=%d, err=%s\n", dev.device_type,
                    cudaGetErrorString(status));
            return status;
        }
        return 0;
    case kDLCPU:
        *out_ptr = malloc(num_bytes);
        break;
    default:
        break;
    }
    return 0;
}

int memory_free(DLDevice dev, void *ptr) {
    int status;
    switch (dev.device_type) {
    case kDLCUDA:
        status = cudaFree(ptr);
        if (unlikely(status)) {
            fprintf(stderr, "free fail, device=%d, err=%s\n", dev.device_type,
                    cudaGetErrorString(status));
            return status;
        }
        return 0;
    case kDLCUDAHost:
        status = cudaFreeHost(ptr);
        if (unlikely(status)) {
            fprintf(stderr, "free fail, device=%d, err=%s\n", dev.device_type,
                    cudaGetErrorString(status));
            return status;
        }
        return 0;
    case kDLCPU:
        free(ptr);
        break;
    default:
        break;
    }
    return 0;
}
