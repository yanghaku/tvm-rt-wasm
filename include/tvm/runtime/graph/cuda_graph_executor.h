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
 * \file runtime/graph_executor/cuda_graph_executor.h
 * \brief cuda_graph_executor struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_CUDA_GRAPH_EXECUTOR_H
#define TVM_RT_CUDA_GRAPH_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph/graph_executor.h>
#include <tvm/runtime/utils/cuda_common.h>

typedef struct CUDAGraphExecutor {
    GRAPH_BASE_MEMBER
    /*! \brief The CUDA stream on which to capture a CUDA graph. */
    CUstream cu_stream;
    /*! \brief The captured CUDA graph will be instantiated to this. */
    CUgraphExec cu_graph_exec;
} CUDAGraphExecutor;

/*!
 * \brief Allocate a new GraphExecutorManager and initialize it with CUDAGraphExecutor.
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
TVM_DLL int CUDAGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                    uint32_t num_dev, GraphExecutorManager **g);

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRun(GraphExecutorManager *g);

/*!
 * \brief Release memory associated with the GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRelease(GraphExecutorManager **g);

/*!
 * \brief Clone a new instance of GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int CUDAGraphExecutorClone(GraphExecutorManager *g, GraphExecutorManager **cloned);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CUDA_GRAPH_EXECUTOR_H
