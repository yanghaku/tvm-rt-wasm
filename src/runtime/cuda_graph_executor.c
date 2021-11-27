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
 * \file src/runtime/cuda_graph_executor.c
 * \brief the implement for cuda_graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/internal/cuda/cuda_graph_executor.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/json.h>
#include <tvm/internal/utils/trie.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_manager_interface.h>

int TVMGraphExecutorCUDACreate(const char *sym_json, TVMModuleHandle module_handle, const DLDevice *devices,
                               uint32_t num_dev, GraphManagerInterface **g) {
    DLDevice cpu = {kDLCPU, 0};
    memory_alloc(sizeof(GraphManagerInterface), cpu, (void **)g);

    (*g)->GetNumOfNodes = GraphExecutorGetNumOfNodes;
    (*g)->GetNodeName = GraphExecutorGetNodeName;
    (*g)->GetInputIndex = GraphExecutorGetInputIndex;
    (*g)->GetOutputIndex = GraphExecutorGetOutputIndex;
    (*g)->GetNumInputs = GraphExecutorGetNumInputs;
    (*g)->GetNumOutputs = GraphExecutorGetNumOutputs;
    (*g)->SetInput = GraphExecutorSetInput;
    (*g)->GetOutput = GraphExecutorGetOutput;
    (*g)->LoadParams = GraphExecutorLoadParams;

    (*g)->Run = CUDAGraphExecutorRun;
    (*g)->Release = CUDAGraphExecutorRelease;
    (*g)->Clone = CUDAGraphExecutorClone;

    memory_alloc(sizeof(CUDAGraphExecutor), cpu, &(*g)->graphHandle);
    return CUDAGraphExecutorLoad(sym_json, module_handle, devices, (*g)->graphHandle);
}

int CUDAGraphExecutorLoad(const char *sym_json, TVMModuleHandle module_handle, const DLDevice *devices,
                          CUDAGraphExecutor *executor) {}

int CUDAGraphExecutorRun(GraphManagerInterface *g) {}

int CUDAGraphExecutorRelease(GraphManagerInterface **g) {}

int CUDAGraphExecutorClone(GraphManagerInterface *g, GraphManagerInterface **cloned) {}
