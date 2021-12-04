/*!
 * \file src/runtime/graph/graph_executor_manager.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/graph/cuda_graph_executor.h>
#include <tvm/runtime/module/module.h>

/*!
 * \brief Create a GraphExecutorManager Instance for given type name
 * @param type graph type name
 * @param graph_json the json for graph
 * @param module_handle TVM Module that exposes the functions to call.
 *
 * \note if the module_handle is NULL, the manager Factory will use the default Module: SystemLib
 *
 * @param devices runtime execution device.
 * @param num_dev the number of devices
 * @param g Pointer which receives a pointer to the newly-created instance.
 * @return 0 if successful
 */
TVM_DLL int GraphExecutorManagerFactory(GraphExecutorType type, const char *graph_json, TVMModuleHandle module_handle,
                                        const DLDevice *devices, uint32_t num_dev, GraphExecutorManager **g) {

    // if module_handle is NULL, we use the systemLib
    if (unlikely(module_handle == NULL)) {
        int status = ModuleFactory(MODULE_SYSTEM_LIB, NULL, 0, (Module **)&module_handle);
        if (unlikely(status)) {
            return status;
        }
    }

    switch (type) {
    case graphExecutor:
        return GraphExecutorCreate(graph_json, module_handle, devices, num_dev, g);
    case graphExecutorCUDA:
        return CUDAGraphExecutorCreate(graph_json, module_handle, devices, num_dev, g);

    default:
        SET_ERROR_RETURN(-1, "unsupported graph executor type");
    }
}
