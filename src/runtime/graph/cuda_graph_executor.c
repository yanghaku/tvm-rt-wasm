/*!
 * \file src/runtime/graph/cuda_graph_executor.c
 * \brief the implement for cuda_graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/device/cpu_memory.h>
#include <tvm/runtime/device/device_api.h>
#include <tvm/runtime/graph/cuda_graph_executor.h>
#include <tvm/runtime/module/module.h>

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
int TVM_RT_WASM_CUDAGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                        uint32_t num_dev, GraphExecutorManager **g) {

#if USE_CUDA // USE_CUDA = 1

    *g = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorManager));

    (*g)->GetNumOfNodes = TVM_RT_WASM_GraphExecutorGetNumOfNodes;
    (*g)->GetNodeName = TVM_RT_WASM_GraphExecutorGetNodeName;
    (*g)->GetInputIndex = TVM_RT_WASM_GraphExecutorGetInputIndex;
    (*g)->GetOutputIndex = TVM_RT_WASM_GraphExecutorGetOutputIndex;
    (*g)->GetNumInputs = TVM_RT_WASM_GraphExecutorGetNumInputs;
    (*g)->GetNumOutputs = TVM_RT_WASM_GraphExecutorGetNumOutputs;
    (*g)->SetInput = TVM_RT_WASM_GraphExecutorSetInput;
    (*g)->SetInputByName = TVM_RT_WASM_GraphExecutorSetInputByName;
    (*g)->GetOutput = TVM_RT_WASM_GraphExecutorGetOutput;
    (*g)->LoadParams = TVM_RT_WASM_GraphExecutorLoadParams;

    (*g)->Run = TVM_RT_WASM_CUDAGraphExecutorRun;
    (*g)->Release = TVM_RT_WASM_CUDAGraphExecutorRelease;
    (*g)->Clone = TVM_RT_WASM_CUDAGraphExecutorClone;

    (*g)->graphHandle = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAGraphExecutor));
    memset((*g)->graphHandle, 0, sizeof(CUDAGraphExecutor));
    return TVM_RT_WASM_GraphExecutorLoad(graph_json, module_handle, devices, num_dev, (*g)->graphHandle);

#else  // USE_CUDA = 0
    CUDA_NOT_SUPPORTED();
#endif // USE_CUDA
}

#if USE_CUDA // USE_CUDA = 1

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_CUDAGraphExecutorRun(GraphExecutorManager *g) {
    //    CHECK_GraphExecutorManager(g);
    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)g->graphHandle;

#ifndef CUDA_10_ONLY
    // init context and stream
    CUDA_DRIVER_CALL(cuStreamCreate(&graph->cu_stream, CU_STREAM_DEFAULT));
    DeviceAPI *deviceApi;
    int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &deviceApi);
    if (unlikely(status)) {
        return status;
    }
    deviceApi->SetStream(graph->devices[0].device_id, graph->cu_stream);

    // begin capture
    CUDA_DRIVER_CALL(cuStreamBeginCapture(graph->cu_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
#endif

    for (uint32_t i = 0; i < graph->num_nodes; ++i) {
        PackedFunction *pf = graph->nodeOps[i].exec;
        int res;
        if (pf) { // call function handle
            if (unlikely(res = pf->exec(graph->nodeOps[i].arg_values, graph->nodeOps[i].arg_type_codes,
                                        graph->nodeOps[i].num_args, &graph->nodeOps[i].return_value,
                                        &graph->nodeOps[i].return_type_code, pf))) {
                return res;
            }
        }
    }

#ifndef CUDA_10_ONLY
    // end capture
    CUgraph cu_graph;
    CUDA_DRIVER_CALL(cuStreamEndCapture(graph->cu_stream, &cu_graph));

    //    size_t numNodes = 0;
    //    CUDA_DRIVER_CALL(cuGraphGetNodes(cu_graph, NULL, &numNodes));
    //    fprintf(stderr, "Num of nodes in the cuda graph created using stream capture API = %zu\n", numNodes);

    // instantiate cuda graph executor
    CUDA_DRIVER_CALL(cuGraphInstantiate(&graph->cu_graph_exec, cu_graph, NULL, NULL, 0));

    // run cuda graph
    CUDA_DRIVER_CALL(cuGraphLaunch(graph->cu_graph_exec, graph->cu_stream));
    CUDA_DRIVER_CALL(cuStreamSynchronize(graph->cu_stream));

#else
    CUDA_DRIVER_CALL(cuStreamSynchronize(NULL));
#endif

    return 0;
}

/*!
 * \brief Release memory associated with the GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_CUDAGraphExecutorRelease(GraphExecutorManager **g) {
    if (unlikely(g == NULL)) {
        SET_ERROR_RETURN(-1, "invalid param: the GraphExecutorManager pointer cannot be NULL");
    }
    CHECK_GraphExecutorManager(*g);

#ifndef CUDA_10_ONLY
    CUDAGraphExecutor *graph = (CUDAGraphExecutor *)((*g)->graphHandle);
    // release cuda special element
    if (graph->cu_graph_exec) {
        CUDA_DRIVER_CALL(cuGraphExecDestroy(graph->cu_graph_exec));
    }
    if (graph->cu_stream) {
        CUDA_DRIVER_CALL(cuStreamDestroy(graph->cu_stream));
    }
#endif

    return TVM_RT_WASM_GraphExecutorRelease(g);
}

/*!
 * \brief Clone a new instance of GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int TVM_RT_WASM_CUDAGraphExecutorClone(GraphExecutorManager *g, GraphExecutorManager **cloned) {
    if (unlikely(cloned == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: the cloned pointer cannot be NULL");
    }
    CHECK_GraphExecutorManager(g);

    // deep copy
    int status = TVM_RT_WASM_GraphExecutorClone(g, cloned);

    // copy to cuda Graph
    GraphExecutor *new_graph = (GraphExecutor *)(*cloned)->graphHandle;
    (*cloned)->graphHandle = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAGraphExecutor));

    CUDAGraphExecutor *new_cu_graph = (CUDAGraphExecutor *)(*cloned)->graphHandle;
    memcpy(new_cu_graph, new_graph, sizeof(GraphExecutor));

#ifndef CUDA_10_ONLY
    new_cu_graph->cu_stream = NULL;
    new_cu_graph->cu_graph_exec = NULL;
#endif

    (*cloned)->Run = TVM_RT_WASM_CUDAGraphExecutorRun;
    (*cloned)->Release = TVM_RT_WASM_CUDAGraphExecutorRelease;
    (*cloned)->Clone = TVM_RT_WASM_CUDAGraphExecutorClone;

    TVM_RT_WASM_HeapMemoryFree(new_graph);
    return status;
}

#endif // USE_CUDA
