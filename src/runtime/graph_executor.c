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
 * \file src/runtime/graph_executor.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/internal/graph/graph_executor.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/json.h>
#include <tvm/internal/utils/trie.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_manager_interface.h>

int TVMGraphExecutorCreate(const char *sym_json, TVMModuleHandle module_handle,
                           const DLDevice *devices, GraphManagerInterface **g) {
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
    (*g)->Run = GraphExecutorRun;
    (*g)->Release = GraphExecutorRelease;
    (*g)->Clone = GraphExecutorClone;

    memory_alloc(sizeof(GraphExecutor), cpu, &(*g)->graphHandle);
    return GraphExecutorLoad(sym_json, module_handle, devices, (*g)->graphHandle);
}

int GraphExecutorLoad(const char *sym_json, TVMModuleHandle module_handle, const DLDevice *devices,
                      GraphExecutor *executor) {}

int GraphExecutorGetNumOfNodes(GraphManagerInterface *g) {}

const char *GraphExecutorGetNodeName(GraphManagerInterface *g, uint32_t nid) {}

int GraphExecutorGetInputIndex(GraphManagerInterface *g, const char *name) {}

int GraphExecutorGetOutputIndex(GraphManagerInterface *g, const char *name) {}

int GraphExecutorGetNumInputs(GraphManagerInterface *g) {}

int GraphExecutorGetNumOutputs(GraphManagerInterface *g) {}

void GraphExecutorSetInput(GraphManagerInterface *g, uint32_t index, const DLTensor *data_in) {}

int GraphExecutorGetOutput(GraphManagerInterface *g, uint32_t index, DLTensor *data_out) {}

int GraphExecutorLoadParams(GraphManagerInterface *g, const char *param_blob, uint32_t param_size) {
}

void GraphExecutorRun(GraphManagerInterface *g) {}

int GraphExecutorRelease(GraphManagerInterface **g) {}

int GraphExecutorClone(GraphManagerInterface *g, GraphManagerInterface **cloned) {}
