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

#include <string.h>
#include <tvm/internal/cuda/cuda_graph_executor.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/json.h>

/*!
 * \brief Allocate a new GraphManagerInterface and initialize it with CUDAGraphExecutor.
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
int TVMGraphExecutorCUDACreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
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
    memset((*g)->graphHandle, 0, sizeof(CUDAGraphExecutor));
    return GraphExecutorLoad(graph_json, module_handle, devices, num_dev, (*g)->graphHandle);
}

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRun(GraphManagerInterface *g) {
    CHECK_GraphManagerInterface(g);
    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)g->graphHandle;

    // init context and stream
    // todo: init context and save stream
    CUDA_DRIVER_CALL(cuStreamCreate(&graph->cu_stream, CU_STREAM_DEFAULT));

    // begin capture
    CUDA_DRIVER_CALL(cuStreamBeginCapture(graph->cu_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));

    for (uint32_t i = 0; i < graph->num_nodes; ++i) {
        if (graph->nodeOps[i].exec) { // call backend function
            graph->nodeOps[i].exec(graph->nodeOps[i].arg_values, graph->nodeOps[i].arg_type_codes,
                                   graph->nodeOps[i].num_args, &graph->nodeOps[i].return_value,
                                   &graph->nodeOps[i].return_type_code, graph->module_handle);
        }
    }

    // end capture
    CUgraph cu_graph;
    CUDA_DRIVER_CALL(cuStreamEndCapture(graph->cu_stream, &cu_graph));
    size_t numNodes = 0;
    CUDA_DRIVER_CALL(cuGraphGetNodes(cu_graph, NULL, &numNodes));
    fprintf(stderr, "Num of nodes in the cuda graph created using stream capture API = %zu\n", numNodes);

    // instantiate cuda graph executor
    CUDA_DRIVER_CALL(cuGraphInstantiate(&graph->cu_graph_exec, cu_graph, NULL, NULL, 0));

    // run cuda graph
    CUDA_DRIVER_CALL(cuGraphLaunch(graph->cu_graph_exec, graph->cu_stream));
    CUDA_DRIVER_CALL(cuStreamSynchronize(graph->cu_stream));

    return 0;
}

/*!
 * \brief Release memory associated with the GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRelease(GraphManagerInterface **g) {
    if (unlikely(g == NULL)) {
        SET_ERROR_RETURN(-1, "invalid param: the graphManagerInterface pointer cannot be NULL");
    }
    CHECK_GraphManagerInterface(*g);

    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)((*g)->graphHandle);
    // release cuda special element
    CUDA_DRIVER_CALL(cuGraphExecDestroy(graph->cu_graph_exec));
    CUDA_DRIVER_CALL(cuStreamDestroy(graph->cu_stream));

    return GraphExecutorRelease(g);
}

/*!
 * \brief Clone a new instance of GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int CUDAGraphExecutorClone(GraphManagerInterface *g, GraphManagerInterface **cloned) {
    if (unlikely(cloned == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: the cloned pointer cannot be NULL");
    }
    CHECK_GraphManagerInterface(g);

    // deep copy
    int status = GraphExecutorClone(g, cloned);

    DLDevice cpu = {kDLCPU, 0};

    // copy to cuda Graph
    GraphExecutor *new_graph = (GraphExecutor *)(*cloned)->graphHandle;
    memory_alloc(sizeof(CUDAGraphExecutor), cpu, &(*cloned)->graphHandle);

    CUDAGraphExecutor *new_cu_graph = (CUDAGraphExecutor *)(*cloned)->graphHandle;
    memcpy(new_cu_graph, new_graph, sizeof(GraphExecutor));
    new_cu_graph->cu_stream = NULL;
    new_cu_graph->cu_graph_exec = NULL;

    memory_free(cpu, new_graph);
    return status;
}
