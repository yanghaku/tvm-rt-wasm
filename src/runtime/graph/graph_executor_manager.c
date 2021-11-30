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
 * \file src/runtime/graph/graph_executor_manager.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>
#include <tvm/runtime/graph/graph_executor.h>
#include <tvm/runtime/utils/common.h>

#if USE_CUDA // USE_CUDA = 1
#include <tvm/runtime/graph/cuda_graph_executor.h>
#endif // USE_CUDA

/*!
 * \brief Create a GraphExecutorManager Instance for given type name
 * @param graph_name graph type name
 * @param graph_json the json for graph
 * @param module_handle TVM Module that exposes the functions to call.
 * @param devices runtime execution device.
 * @param num_dev the number of devices
 * @param g Pointer which receives a pointer to the newly-created instance.
 * @return 0 if successful
 */
TVM_DLL int GraphExecutorManagerFactory(const char *graph_name, const char *graph_json, TVMModuleHandle module_handle,
                                        const DLDevice *devices, uint32_t num_dev, GraphExecutorManager **g) {
    if (strcmp(graph_name, "graph_executor") == 0) {
        return GraphExecutorCreate(graph_json, module_handle, devices, num_dev, g);
    }

#if USE_CUDA // USE_CUDA = 1
    else if (strcmp(graph_name, "cuda_graph_executor") == 0) {
        return CUDAGraphExecutorCreate(graph_json, module_handle, devices, num_dev, g);
    }
#endif

    else {
        SET_ERROR_RETURN(-1, "unsupported graph executor name: %s", graph_name);
    }
}
