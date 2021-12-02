/*!
 * \file src/runtime/graph/cuda_graph_executor.c
 * \brief the implement for cuda_graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_executor_manager.h>
#include <tvm/runtime/utils/common.h>
#include <tvm/runtime/utils/json.h>

#if USE_CUDA
#include <tvm/runtime/graph/cuda_graph_executor.h>
#endif

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
int CUDAGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                            uint32_t num_dev, GraphExecutorManager **g) {

#if USE_CUDA // USE_CUDA = 1

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(GraphExecutorManager), 0, no_type, (void **)&g);

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

    TVMDeviceAllocDataSpace(cpu, sizeof(CUDAGraphExecutor), 0, no_type, &(*g)->graphHandle);
    memset((*g)->graphHandle, 0, sizeof(CUDAGraphExecutor));
    return GraphExecutorLoad(graph_json, module_handle, devices, num_dev, (*g)->graphHandle);

#else  // USE_CUDA = 0
    fprintf(stderr, "CUDA library is not supported! you can compile from source and set USE_CUDA option ON\n");
    exit(-1);
#endif // USE_CUDA
}

#if USE_CUDA // USE_CUDA = 1

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRun(GraphExecutorManager *g) {
    CHECK_GraphExecutorManager(g);
    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)g->graphHandle;

    // init context and stream
    // todo: init context and save stream
    CUDA_DRIVER_CALL(cuStreamCreate(&graph->cu_stream, CU_STREAM_DEFAULT));

    // begin capture
    CUDA_DRIVER_CALL(cuStreamBeginCapture(graph->cu_stream, CU_STREAM_CAPTURE_MODE_GLOBAL));

    for (uint32_t i = 0; i < graph->num_nodes; ++i) {
        if (graph->nodeOps[i].exec) { // run function handle
            TVMFuncCall(graph->nodeOps[i].exec, graph->nodeOps[i].arg_values, graph->nodeOps[i].arg_type_codes,
                        graph->nodeOps[i].num_args, &graph->nodeOps[i].return_value,
                        &graph->nodeOps[i].return_type_code);
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
 * \brief Release memory associated with the GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int CUDAGraphExecutorRelease(GraphExecutorManager **g) {
    if (unlikely(g == NULL)) {
        SET_ERROR_RETURN(-1, "invalid param: the GraphExecutorManager pointer cannot be NULL");
    }
    CHECK_GraphExecutorManager(*g);

    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)((*g)->graphHandle);
    // release cuda special element
    CUDA_DRIVER_CALL(cuGraphExecDestroy(graph->cu_graph_exec));
    CUDA_DRIVER_CALL(cuStreamDestroy(graph->cu_stream));

    return GraphExecutorRelease(g);
}

/*!
 * \brief Clone a new instance of GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int CUDAGraphExecutorClone(GraphExecutorManager *g, GraphExecutorManager **cloned) {
    if (unlikely(cloned == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: the cloned pointer cannot be NULL");
    }
    CHECK_GraphExecutorManager(g);

    // deep copy
    int status = GraphExecutorClone(g, cloned);

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};

    // copy to cuda Graph
    GraphExecutor *new_graph = (GraphExecutor *)(*cloned)->graphHandle;
    TVMDeviceAllocDataSpace(cpu, sizeof(CUDAGraphExecutor), 0, no_type, &(*cloned)->graphHandle);

    CUDAGraphExecutor *new_cu_graph = (CUDAGraphExecutor *)(*cloned)->graphHandle;
    memcpy(new_cu_graph, new_graph, sizeof(GraphExecutor));
    new_cu_graph->cu_stream = NULL;
    new_cu_graph->cu_graph_exec = NULL;

    (*cloned)->Run = CUDAGraphExecutorRun;
    (*cloned)->Release = CUDAGraphExecutorRelease;
    (*cloned)->Clone = CUDAGraphExecutorClone;

    TVMDeviceFreeDataSpace(cpu, new_graph);
    return status;
}

#endif // USE_CUDA
