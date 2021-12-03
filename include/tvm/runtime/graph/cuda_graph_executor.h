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

#if USE_CUDA // USE_CUDA = 1

    /*! \brief The CUDA stream on which to capture a CUDA graph. */
    CUstream cu_stream;
    /*! \brief The captured CUDA graph will be instantiated to this. */
    CUgraphExec cu_graph_exec;

#endif // USE_CUDA
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
int CUDAGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
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
