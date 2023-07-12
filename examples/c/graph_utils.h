#ifndef TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
#define TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H

#include "dlpack/dlpack.h"
#include "during.h"
#include "graph_executor.h"
#include "tvm_error_process.h"

typedef TVM_RT_WASM_GraphExecutor GraphHandle;

int init_graph(TVMModuleHandle module, const char *graph_param_path, const char *graph_json,
               GraphHandle *graph_handle_ptr) {
    int status;
    SET_TIME(t0) // init graph start

#if EXAMPLE_USE_CUDA
    DLDevice graph_device = {kDLCUDA, 0};
#elif EXAMPLE_USE_WEBGPU
    DLDevice graph_device = {kDLWebGPU, 0};
#else
    DLDevice graph_device = {kDLCPU, 0};
#endif
    *graph_handle_ptr = TVM_RT_WASM_GraphExecutorCreate(graph_json, module, &graph_device, 1);

    if (!*graph_handle_ptr) {
        return -1;
    }

    SET_TIME(t1) // init graph end, load params start

    RUN(TVM_RT_WASM_GraphExecutorLoadParamsFromFile(*graph_handle_ptr, graph_param_path));

    SET_TIME(t2) // load graph end, set input start

    printf("Create graph time: %lf ms\nLoad graph params time: %lf ms\n\n", GET_DURING(t1, t0),
           GET_DURING(t2, t1));
    return status;
}

int init_graph_with_dso_lib(const char *module_filename, const char *graph_param_path,
                            const char *graph_json, GraphHandle *graph_handle_ptr) {
    TVMModuleHandle module = NULL;
    int status = TVMModLoadFromFile(module_filename, "so", &module);
    if (status) {
        return status;
    }
    status = init_graph(module, graph_param_path, graph_json, graph_handle_ptr);
    if (status) {
        // if create fail, the module need to be freed.
        TVMModFree(module);
    }
    return status;
}

int init_graph_with_syslib(const char *graph_param_path, const char *graph_json,
                           GraphHandle *graph_handle_ptr) {
    TVMModuleHandle syslib = NULL;
    return init_graph(syslib, graph_param_path, graph_json, graph_handle_ptr);
}

int run_graph(GraphHandle graph_handle, const DLTensor *inputs, const char **input_names,
              int input_num, DLTensor *outputs, const int *output_indexes, int output_num) {
    int status;

    SET_TIME(t0) // set input start

    for (int i = 0; i < input_num; ++i) {
        RUN(TVM_RT_WASM_GraphExecutorSetInputByName(graph_handle, input_names[i], inputs + i));
    }

    SET_TIME(t1) // set input end, run graph start

    RUN(TVM_RT_WASM_GraphExecutorRun(graph_handle));

    SET_TIME(t2) // run end, get output start

    for (int i = 0; i < output_num; ++i) {
        RUN(TVM_RT_WASM_GraphExecutorGetOutput(graph_handle, output_indexes[i], outputs + i));
    }

    SET_TIME(t3) // get output end
    printf("Set graph input time: %lf ms\nRun graph time: %lf ms\nGet graph output time: %lf ms\n",
           GET_DURING(t1, t0), GET_DURING(t2, t1), GET_DURING(t3, t2));
    return status;
}

inline int delete_graph(GraphHandle graph_handle) {
    return TVM_RT_WASM_GraphExecutorFree(graph_handle);
}

#endif // TVM_RT_EXAMPLE_WASM_GRAPH_UTILS_H
