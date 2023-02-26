#ifndef TVM_RT_WASM_CMP_NATIVE_GRAPH_UTILS_H
#define TVM_RT_WASM_CMP_NATIVE_GRAPH_UTILS_H

#include "dlpack/dlpack.h"
#include "during.h"
#include "tvm_error_process.h"
#include <stdlib.h>
#include <tvm/runtime/c_runtime_api.h>

// use mmap for load params
#if defined(__unix) || defined(__unix__)
#define MMAP
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#endif

typedef TVMModuleHandle GraphHandle;

#define SYS_LIB_NAME "runtime.SystemLib"
#define CUDA_GRAPH_CREATE_NAME "tvm.graph_executor.create"
#define GRAPH_LOAD_PARAMS_NAME "load_params"
#define GRAPH_SET_INPUT_NAME "set_input"
#define GRAPH_RUN_NAME "run"
#define GRAPH_GET_OUTPUT_NAME "get_output"

#define CHECK_FUNC_FROM_MODULE(f, name)                                                                                \
    do {                                                                                                               \
        if ((f) == NULL) {                                                                                             \
            fprintf(stderr, "Cannot get function `%s` from graph\n", (name));                                          \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

/// use a created stream instead of default stream, avoid block with other instance.
static TVMStreamHandle cuda_stream = NULL;

int init_graph_with_syslib(const char *graph_param_path, const char *graph_json, GraphHandle *graph_handle_ptr) {
    int status;
    if (cuda_stream == NULL) {
        RUN(TVMStreamCreate(kDLCUDA, 0, &cuda_stream));
        RUN(TVMSetStream(kDLCUDA, 0, cuda_stream));
    }

    TVMValue tvm_ret = {.v_handle = NULL};
    TVMArgTypeCode tvm_ret_code = 0;
    *graph_handle_ptr = NULL;

    SET_TIME(t0) // init graph start

    // create system library module
    TVMFunctionHandle sys_lib_create_func = NULL;
    RUN(TVMFuncGetGlobal(SYS_LIB_NAME, &sys_lib_create_func));
    if (sys_lib_create_func == NULL) {
        fprintf(stderr, "Cannot get `%s` symbol from tvm runtime\n", SYS_LIB_NAME);
        return -1;
    }
    RUN(TVMFuncCall(sys_lib_create_func, NULL, NULL, 0, &tvm_ret, &tvm_ret_code));
    if (tvm_ret_code != kTVMModuleHandle || tvm_ret.v_handle == NULL) {
        fprintf(stderr, "Create system library module fail\n");
        return -1;
    }
    TVMModuleHandle sys_lib = tvm_ret.v_handle;

    // create graph
    TVMFunctionHandle graph_create_func = NULL;
    RUN(TVMFuncGetGlobal(CUDA_GRAPH_CREATE_NAME, &graph_create_func));
    if (graph_create_func == NULL) {
        fprintf(stderr, "Cannot get `%s` from tvm runtime, you can set `USE_GRAPH_EXECUTOR=ON` to recompile tvm\n",
                CUDA_GRAPH_CREATE_NAME);
        return -1;
    }

    TVMValue create_args[] = {{.v_str = graph_json}, {.v_handle = sys_lib}, {.v_int64 = kDLCUDA}, {.v_int64 = 0}};
    TVMArgTypeCode create_type_code[] = {kTVMStr, kTVMModuleHandle, kDLInt, kDLInt};
    tvm_ret.v_handle = NULL;
    tvm_ret_code = 0;
    RUN(TVMFuncCall(graph_create_func, create_args, create_type_code, 4, &tvm_ret, &tvm_ret_code));
    if (tvm_ret_code != kTVMModuleHandle || tvm_ret.v_handle == NULL) {
        fprintf(stderr, "Create CUDA Graph Executor Module Fail\n");
        return -1;
    }
    GraphHandle graph_handle = tvm_ret.v_handle;

    SET_TIME(t1) // init graph end, load params start

    TVMPackedCFunc load_params_func = NULL;
    RUN(TVMModGetFunction(graph_handle, GRAPH_LOAD_PARAMS_NAME, 0, &load_params_func));
    CHECK_FUNC_FROM_MODULE(load_params_func, GRAPH_LOAD_PARAMS_NAME);

#ifdef MMAP
    int fd = open(graph_param_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot read file from %s\n", graph_param_path);
        return -1;
    }
    struct stat file_info;
    stat(graph_param_path, &file_info);
    char *param_blob = (char *)mmap(NULL, file_info.st_size, PROT_READ, MAP_SHARED | PROT_READ, fd, 0);
    if (param_blob == MAP_FAILED) {
        fprintf(stderr, "mmap fail!\n");
        return -1;
    }
    close(fd);
    TVMByteArray params = {.data = param_blob, .size = file_info.st_size};
#else
    FILE *f = fopen(graph_param_path, "rb");
    size_t f_size = 0;
    if (f != NULL) {
        if (!fseek(f, 0, SEEK_END)) {
            f_size = ftell(f);
            fseek(f, 0, SEEK_SET);
        }
    }
    if (f_size == 0) {
        if (f != NULL) {
            fclose(f);
        }
        fprintf(stderr, "Cannot read file from %s\n", graph_param_path);
        return -1;
    }
    char *param_blob = (char *)malloc(f_size + 1);
    fread(param_blob, 1, f_size, f);
    fclose(f);
    TVMByteArray params = {.data = param_blob, .size = f_size};
#endif // MMAP
    TVMValue arg[] = {{.v_handle = &params}};
    TVMArgTypeCode tp[] = {kTVMBytes};
    RUN(TVMFuncCall(load_params_func, arg, tp, 1, &tvm_ret, &tvm_ret_code));

    SET_TIME(t2) // load graph end, set input start

    printf("Create graph time: %lf ms\nLoad graph params time: %lf ms\n\n", GET_DURING(t1, t0), GET_DURING(t2, t1));
    *graph_handle_ptr = graph_handle;
    return status;
}

int run_graph(GraphHandle graph_handle, const DLTensor *inputs, const char **input_names, int input_num,
              const DLTensor *outputs, const int *output_indexes, int output_num) {
    int status;
    TVMValue tvm_ret;
    TVMArgTypeCode tvm_ret_code;

    SET_TIME(t0) // set input start

    TVMPackedCFunc set_input_func = NULL;
    RUN(TVMModGetFunction(graph_handle, GRAPH_SET_INPUT_NAME, 0, &set_input_func));
    CHECK_FUNC_FROM_MODULE(set_input_func, GRAPH_SET_INPUT_NAME);

    for (int i = 0; i < input_num; ++i) {
        TVMValue args[] = {
            {.v_str = input_names[i]},
            {.v_handle = inputs + i},
        };
        TVMArgTypeCode arg_types[] = {kTVMStr, kTVMDLTensorHandle};
        RUN(TVMFuncCall(set_input_func, args, arg_types, 2, &tvm_ret, &tvm_ret_code));
    }
    RUN(TVMFuncFree(set_input_func));

    SET_TIME(t1) // set input end, run graph start

    TVMPackedCFunc run_func = NULL;
    RUN(TVMModGetFunction(graph_handle, GRAPH_RUN_NAME, 0, &run_func));
    CHECK_FUNC_FROM_MODULE(set_input_func, GRAPH_SET_INPUT_NAME);
    RUN(TVMFuncCall(run_func, NULL, NULL, 0, &tvm_ret, &tvm_ret_code));
    RUN(TVMFuncFree(run_func));

    RUN(TVMSynchronize(kDLCUDA, 0, cuda_stream));
    SET_TIME(t2) // run end, get output start

    TVMPackedCFunc get_output_func = NULL;
    RUN(TVMModGetFunction(graph_handle, GRAPH_GET_OUTPUT_NAME, 0, &get_output_func));
    CHECK_FUNC_FROM_MODULE(get_output_func, GRAPH_GET_OUTPUT_NAME);
    for (int i = 0; i < output_num; ++i) {
        TVMValue args[] = {
            {.v_int64 = output_indexes[i]},
            {.v_handle = outputs + i},
        };
        TVMArgTypeCode arg_types[] = {kTVMArgInt, kTVMDLTensorHandle};
        RUN(TVMFuncCall(get_output_func, args, arg_types, 2, &tvm_ret, &tvm_ret_code));
    }

    SET_TIME(t3) // get output end
    printf("Set graph input time: %lf ms\nRun graph time: %lf ms\nGet graph output time: %lf ms\n", GET_DURING(t1, t0),
           GET_DURING(t2, t1), GET_DURING(t3, t2));
    return status;
}

inline int delete_graph(GraphHandle graph_handle) { return TVMModFree(graph_handle); }

#endif // TVM_RT_WASM_CMP_NATIVE_GRAPH_UTILS_H
