#ifndef TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
#define TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H

#include "dlpack/dlpack.h"
#include "during.h"
#include "graph_executor.h"
#include "tvm_error_process.h"

typedef TVM_RT_WASM_GraphExecutor GraphHandle;

int init_graph_with_syslib(const char *graph_param_path, const char *graph_json, GraphHandle *graph_handle_ptr) {
    TVMModuleHandle syslib = NULL;
    int status;

    SET_TIME(t0) // init graph start

#if EXAMPLE_USE_CUDA
    DLDevice cuda = {kDLCUDA, 0};
    RUN(TVM_RT_WASM_GraphExecutorCreate(graph_json, syslib, &cuda, 1, graph_handle_ptr));
#else
    DLDevice cpu = {kDLCPU, 0};
    RUN(TVM_RT_WASM_GraphExecutorCreate(graph_json, syslib, &cpu, 1, graph_handle_ptr));
#endif

    SET_TIME(t1) // init graph end, load params start

    RUN(TVM_RT_WASM_GraphExecutorLoadParamsFromFile(*graph_handle_ptr, graph_param_path));

    SET_TIME(t2) // load graph end, set input start

    printf("Create graph time: %lf ms\nLoad graph params time: %lf ms\n\n", GET_DURING(t1, t0), GET_DURING(t2, t1));
    return status;
}

int run_graph(GraphHandle graph_handle, const DLTensor *inputs, const char **input_names, int input_num,
              const DLTensor *outputs, const int *output_indexes, int output_num) {
    int status;

    SET_TIME(t0) // set input start

    for (int i = 0; i < input_num; ++i) {
        RUN(TVM_RT_WASM_GraphExecutorSetInput(graph_handle, input_names[i], inputs + i));
    }

    SET_TIME(t1) // set input end, run graph start

    RUN(TVM_RT_WASM_GraphExecutorRun(graph_handle));

    SET_TIME(t2) // run end, get output start

    for (int i = 0; i < output_num; ++i) {
        RUN(TVM_RT_WASM_GraphExecutorGetOutput(graph_handle, output_indexes[i], outputs + i));
    }

    SET_TIME(t3) // get output end
    printf("Set graph input time: %lf ms\nRun graph time: %lf ms\nGet graph output time: %lf ms\n", GET_DURING(t1, t0),
           GET_DURING(t2, t1), GET_DURING(t3, t2));
    return status;
}

inline int delete_graph(GraphHandle graph_handle) { return TVM_RT_WASM_GraphExecutorDestory(&graph_handle); }

#endif // TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
