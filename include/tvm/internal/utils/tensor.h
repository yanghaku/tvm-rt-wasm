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
 * \file internal/utils/tensor.h
 * \brief the utils function for DLTensor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_TENSOR_H
#define TVM_RT_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

/*!
 * \brief copy the data memory for DLTensor
 * @param data_from src DLTensor
 * @param data_to dst DLTensor
 * @param stream stream handle
 * @return 0 if successful
 */
static inline int DLTensor_CopyFromTo(const DLTensor *data_from, DLTensor *data_to, TVMStreamHandle *stream);

/*!
 * \brief parse string to DLDataType
 * @param str the source string
 * @param str_len source string length
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
static inline int DLDataType_ParseFromString(const char *str, uint32_t str_len, DLDataType *out_type);

/*!
 * \brief get data number of Tensor
 * @param shape the shape of tensor
 * @param ndim the number of dim
 * @return result
 */
static inline uint64_t DLTensor_GetDataSize(const uint64_t *shape, int ndim) {
    uint64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

/*!
 * \brief get data bytes number of tensor
 * @param tensor the tensor pointer
 * @return result
 */
static inline uint64_t DLTensor_GetDataBytes(const DLTensor *tensor) {
    size_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        size *= tensor->shape[i];
    }
    size *= (tensor->dtype.bits * tensor->dtype.lanes + 7) / 8;
    return size;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_TENSOR_H
