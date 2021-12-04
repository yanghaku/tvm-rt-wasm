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

#include <assert.h>
#include <dlpack/dlpack.h>
#include <float.h>
#include <stdio.h>
#include <sys/time.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_executor_manager.h>

#define OUTPUT_LEN 1024

extern const unsigned char graph_json[];
// extern const unsigned int graph_json_len;
extern const unsigned char graph_params[];
extern const unsigned int graph_params_len;

#define RUN(func)                                                                                                      \
    do {                                                                                                               \
        status = (func);                                                                                               \
        if (status) {                                                                                                  \
            fprintf(stderr, "%s\n", TVMGetLastError());                                                                \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

extern void *__tvm_module_cxt;
int main(int argc, char **argv) {
    fprintf(stderr, "module ctx = %p\n", &__tvm_module_cxt);
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <picture.bin>\n", __FILE_NAME__);
        return -1;
    }

    // static array storage
    static float input_storage[1 * 3 * 224 * 224];
    static float output_storage[OUTPUT_LEN];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    DLTensor input, output;

    struct timeval t0, t1, t2, t3, t4, t5;
    GraphExecutorManager *graphManager;
    TVMModuleHandle syslib = NULL;
    int status;

    gettimeofday(&t0, 0); // init start

#if EXAMPLE_USE_CUDA
    DLDevice cuda = {kDLCUDA, 0};
    RUN(GraphExecutorManagerFactory(graphExecutorCUDA, (const char *)graph_json, syslib, &cuda, 1, &graphManager));
#else
    RUN(GraphExecutorManagerFactory(graphExecutor, (const char *)graph_json, syslib, &cpu, 1, &graphManager));
#endif

    RUN(graphManager->LoadParams(graphManager, (const char *)graph_params, graph_params_len));

    gettimeofday(&t1, 0); // load graph end, set input start

    // load input from file
    FILE *fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        fprintf(stderr, "cannot open file %s\n", argv[1]);
        return -1;
    }
    fread(input_storage, 3 * 224 * 224, 4, fp);
    fclose(fp);
    input.data = input_storage;
    input.device = cpu;
    input.ndim = 4;
    input.dtype = float32;
    int64_t shape[4] = {1, 3, 224, 224};
    input.shape = shape;
    input.strides = NULL;
    input.byte_offset = 0;

    RUN(graphManager->SetInputByName(graphManager, "data", &input));

    gettimeofday(&t2, 0); // set input end, run graph start

    RUN(graphManager->Run(graphManager));

    gettimeofday(&t3, 0); // run end, get output start

    output.data = output_storage;
    output.device = cpu;
    output.ndim = 2;
    output.dtype = float32;
    int64_t out_shape[2] = {1, 1000};
    output.shape = out_shape;
    output.strides = NULL;
    output.byte_offset = 0;

    RUN(graphManager->GetOutput(graphManager, 0, &output));

    gettimeofday(&t4, 0); // get output end, destroy start

    float max_iter = -FLT_MAX;
    int32_t max_index = -1;
    for (int i = 0; i < OUTPUT_LEN; ++i) {
        if (output_storage[i] > max_iter) {
            max_iter = output_storage[i];
            max_index = i;
        }
    }

    for (int i = 0; i < OUTPUT_LEN; ++i) {
        int s = (int)(output_storage[i] * 10000.0);
        if (s != 0) {
            fprintf(stderr, "%d: %d\n", i, s);
        }
    }
    RUN(graphManager->Release(&graphManager));

    gettimeofday(&t5, 0);

    printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);
    printf("timing: %.2f ms (create), %.2f ms (set_input), %.2f ms (run), "
           "%.2f ms (get_output), %.2f ms (destroy)\n",
           (double)(t1.tv_sec - t0.tv_sec) * 1000 + (double)(t1.tv_usec - t0.tv_usec) / 1000.f,
           (double)(t2.tv_sec - t1.tv_sec) * 1000 + (double)(t2.tv_usec - t1.tv_usec) / 1000.f,
           (double)(t3.tv_sec - t2.tv_sec) * 1000 + (double)(t3.tv_usec - t2.tv_usec) / 1000.f,
           (double)(t4.tv_sec - t3.tv_sec) * 1000 + (double)(t4.tv_usec - t3.tv_usec) / 1000.f,
           (double)(t5.tv_sec - t4.tv_sec) * 1000 + (double)(t5.tv_usec - t4.tv_usec) / 1000.f);

    return 0;
}
