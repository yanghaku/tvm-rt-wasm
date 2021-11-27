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
 * \file internal/cuda/cuda_graph_executor.h
 * \brief cuda_graph_executor struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_CUDA_GRAPH_EXECUTOR_H
#define TVM_RT_CUDA_GRAPH_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/internal/cuda/cuda_common.h>
#include <tvm/internal/graph/graph_executor.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>

typedef struct CUDAGraphExecutor {
    GRAPH_BASE_MEMBER
    /*! \brief The CUDA stream on which to capture a CUDA graph. */
    CUstream cu_stream;
    /*! \brief The captured CUDA graph will be instantiated to this. */
    CUgraphExec cu_graph_exec;
} CUDAGraphExecutor;

/*!
 * \brief init a new CUDAGraphExecutor from graph.json
 *
 * \param sym_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param executor the instance instance.
 * \return 0 if successful.
 */
int CUDAGraphExecutorLoad(const char *sym_json, TVMModuleHandle module_handle, const DLDevice *devices,
                          CUDAGraphExecutor *executor);

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRun(GraphManagerInterface *g);

/*!
 * \brief Release memory associated with the GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRelease(GraphManagerInterface **g);

/*!
 * \brief Clone a new instance of GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int CUDAGraphExecutorClone(GraphManagerInterface *g, GraphManagerInterface **cloned);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_CUDA_GRAPH_EXECUTOR_H
