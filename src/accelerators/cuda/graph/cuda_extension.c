/**
 * @file cuda/graph/cuda_extension.c
 * @brief The implementation for graph_executor cuda extension.
 */

#ifndef CUDA_10_ONLY

#include <cuda_common.h>
#include <device/cpu_memory.h>
#include <device/device_api.h>
#include <graph/graph_executor.h>
#include <string.h>

typedef struct CUDAGraphExecutorExtensionData {
    /** @brief The CUDA stream on which to capture a CUDA graph. */
    CUstream cu_stream;
    /** @brief The captured CUDA graph will be instantiated to this. */
    CUgraphExec cu_graph_exec;
} CUDAGraphExecutorExtensionData;

/**
 * @brief Execute the graph.
 * @param g The instance of This.
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutorCUDARun(TVM_RT_WASM_GraphExecutor graph) {
    CUDAGraphExecutorExtensionData *data = (CUDAGraphExecutorExtensionData *)graph->extension_data;
    // run cuda graph
    CUDA_DRIVER_CALL(cuGraphLaunch(data->cu_graph_exec, data->cu_stream));
    // wait for running
    CUDA_DRIVER_CALL(cuStreamSynchronize(data->cu_stream));
    return 0;
}

/**
 * @brief Release the CUDAGraphExecutorExtensionData.
 * @param d Pointer to CUDAGraphExecutorExtensionData.
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutorCUDAFree(void *d) {
    if (unlikely(d == NULL)) {
        return 0;
    }

    CUDAGraphExecutorExtensionData *data = (CUDAGraphExecutorExtensionData *)d;
    if (data->cu_graph_exec) {
        cuGraphExecDestroy(data->cu_graph_exec);
    }
    if (data->cu_stream) {
        cuStreamDestroy(data->cu_stream);
    }

    TVM_RT_WASM_HeapMemoryFree(d);
    return 0;
}

/**
 * @brief Clone the CUDAGraphExecutorExtensionData.
 * @param d Pointer to CUDAGraphExecutorExtensionData.
 * @param cloned Pointer which receive the new instance.
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutorCUDAClone(void *d, void **cloned) {
    (void)d;
    if (unlikely(cloned == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "The cloned pointer cannot be NULL.");
    }

    // todo
    return 0;
}

/**
 * @brief Allocate and initialize CUDA extension data for TVM_RT_WASM_GraphExecutor.
 *
 * @param g The instance of TVM_RT_WASM_GraphExecutor.
 * @return 0 if successful.
 */
int TVM_RT_WASM_CUDAGraphExecutorExtensionDataCreate(TVM_RT_WASM_GraphExecutor g) {
    CUDAGraphExecutorExtensionData *d =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUDAGraphExecutorExtensionData));
    DeviceAPI *deviceApi = NULL;
    int status = TVM_RT_WASM_DeviceAPIGet(kDLCUDA, &deviceApi);
    if (unlikely(status)) {
        goto fail;
    }

    // init context and stream
    CUDA_DRIVER_CALL_OR_GOTO(cuStreamCreate(&d->cu_stream, CU_STREAM_DEFAULT), fail);
    deviceApi->SetStream(g->devices[0].device_id, d->cu_stream);

    // begin capture
    CUDA_DRIVER_CALL_OR_GOTO(
        cuStreamBeginCapture(d->cu_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), fail);

    status = TVM_RT_WASM_GraphExecutorRun(g);
    if (unlikely(status)) {
        goto fail;
    }

    // end capture
    CUgraph cu_graph;
    CUDA_DRIVER_CALL_OR_GOTO(cuStreamEndCapture(d->cu_stream, &cu_graph), fail);

    // instantiate cuda graph executor
    CUDA_DRIVER_CALL_OR_GOTO(cuGraphInstantiate(&d->cu_graph_exec, cu_graph, 0), fail);

    g->extension_data = (void *)d;
    g->Free = TVM_RT_WASM_GraphExecutorCUDAFree;
    g->Run = TVM_RT_WASM_GraphExecutorCUDARun;
    g->Clone = TVM_RT_WASM_GraphExecutorCUDAClone;

    return status;
fail:
    if (d) {
        TVM_RT_WASM_HeapMemoryFree(d);
    }
    return -1;
}

#endif // !CUDA_10_ONLY
