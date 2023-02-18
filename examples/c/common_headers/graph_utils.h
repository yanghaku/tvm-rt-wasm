#ifndef TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
#define TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H

#include "dlpack/dlpack.h"
#include "during.h"
#include "tvm/runtime/graph_executor_manager.h"
#include "tvm_error_process.h"

int init_graph_with_syslib(const char *graph_param_path, const char *graph_json,
                           GraphExecutorManager **graphManagerPtr) {
    TVMModuleHandle syslib = NULL;
    int status;

    SET_TIME(t0) // init graph start

#if EXAMPLE_USE_CUDA
    DLDevice cuda = {kDLCUDA, 0};
    RUN(GraphExecutorManagerFactory(graphExecutorCUDA, graph_json, syslib, &cuda, 1, graphManagerPtr));
#else
    DLDevice cpu = {kDLCPU, 0};
    RUN(GraphExecutorManagerFactory(graphExecutor, graph_json, syslib, &cpu, 1, graphManagerPtr));
#endif

    SET_TIME(t1) // init graph end, load params start

    RUN((*graphManagerPtr)->LoadParamsFromFile((*graphManagerPtr), graph_param_path));

    SET_TIME(t2) // load graph end, set input start

    printf("Create graph time: %lf ms\nLoad graph params time: %lf ms\n\n", GET_DURING(t1, t0), GET_DURING(t2, t1));
    return status;
}

int run_graph(GraphExecutorManager *graphManager, const DLTensor *input, DLTensor *output, const char *input_name,
              int output_index) {
    int status;

    SET_TIME(t0) // set input start

    RUN(graphManager->SetInputByName(graphManager, input_name, input));

    SET_TIME(t1) // set input end, run graph start

    RUN(graphManager->Run(graphManager));

    SET_TIME(t2) // run end, get output start

    RUN(graphManager->GetOutput(graphManager, output_index, output));

    SET_TIME(t3) // get output end
    printf("Set graph input time: %lf ms\nRun graph time: %lf ms\nGet graph output time: %lf ms\n", GET_DURING(t1, t0),
           GET_DURING(t2, t1), GET_DURING(t3, t2));
    return status;
}

#endif // TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
