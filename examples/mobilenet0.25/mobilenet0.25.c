#ifdef _MSC_VER

#include <Windows.h>

#define SET_TIME(t0)                                                                                                   \
    long long(t0);                                                                                                     \
    GetSystemTimePreciseAsFileTime(&(t0));

#define GET_DURING(t1, t0) (((double)((t1) - (t0))) / 10000.0)

#else

#include <sys/time.h>

#define SET_TIME(t0)                                                                                                   \
    struct timeval(t0);                                                                                                \
    gettimeofday(&(t0), NULL);

#define GET_DURING(t1, t0) ((double)((t1).tv_sec - (t0).tv_sec) * 1000 + (double)((t1).tv_usec - (t0).tv_usec) / 1000.0)

#endif

#include <dlpack/dlpack.h>
#include <float.h>
#include <stdio.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_executor_manager.h>

#define OUTPUT_LEN 1024

#define INPUT_SHAPE (1 * 3 * 224 * 224)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

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

extern void *__tvm_module_ctx;
#if EXAMPLE_USE_CUDA
extern void *__tvm_dev_mblob;
#endif

int main() {
    SET_TIME(start_time)
    fprintf(stderr, "module ctx = %p\n", &__tvm_module_ctx);
#if EXAMPLE_USE_CUDA
    fprintf(stderr, "dev ctx = %p\n", &__tvm_dev_mblob);
#endif

    // static array storage
    static float input_storage[INPUT_SHAPE];
    static float output_storage[OUTPUT_LEN];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    DLTensor input, output;

    GraphExecutorManager *graphManager;
    TVMModuleHandle syslib = NULL;
    int status;

    SET_TIME(t0) // init start

#if EXAMPLE_USE_CUDA
    DLDevice cuda = {kDLCUDA, 0};
    RUN(GraphExecutorManagerFactory(graphExecutorCUDA, (const char *)graph_json, syslib, &cuda, 1, &graphManager));
#else
    RUN(GraphExecutorManagerFactory(graphExecutor, (const char *)graph_json, syslib, &cpu, 1, &graphManager));
#endif

    SET_TIME(t1)

    RUN(graphManager->LoadParams(graphManager, (const char *)graph_params, graph_params_len));

    SET_TIME(t2) // set input start

    // load input from stdin
    size_t len = fread(input_storage, 4, INPUT_SHAPE, stdin);
    if (len != INPUT_SHAPE) {
        fprintf(stderr, INPUT_ERR_MSG);
        return -1;
    }
    input.data = input_storage;
    input.device = cpu;
    input.ndim = 4;
    input.dtype = float32;
    int64_t shape[4] = {1, 3, 224, 224};
    input.shape = shape;
    input.strides = NULL;
    input.byte_offset = 0;

    RUN(graphManager->SetInputByName(graphManager, "data", &input));

    SET_TIME(t3) // set input end, run graph start

    RUN(graphManager->Run(graphManager));

    SET_TIME(t4) // run end, get output start

    output.data = output_storage;
    output.device = cpu;
    output.ndim = 2;
    output.dtype = float32;
    int64_t out_shape[2] = {1, 1000};
    output.shape = out_shape;
    output.strides = NULL;
    output.byte_offset = 0;

    RUN(graphManager->GetOutput(graphManager, 0, &output));

    SET_TIME(t5) // get output end, destroy start

    float max_iter = -FLT_MAX;
    int32_t max_index = -1;
    for (int i = 0; i < OUTPUT_LEN; ++i) {
        if (output_storage[i] > max_iter) {
            max_iter = output_storage[i];
            max_index = i;
        }
    }

    //    for (int i = 0; i < OUTPUT_LEN; ++i) {
    //        int s = (int)(output_storage[i] * 10000.0);
    //        if (s != 0) {
    //            fprintf(stderr, "%d: %d\n", i, s);
    //        }
    //    }
    RUN(graphManager->Release(&graphManager));

    SET_TIME(t6)

    printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);
    printf("create time: %lf ms\nload_params time: %lf ms\nset_input time: %lf\nrun time: %lf ms\nget_output time: %lf "
           "ms\ndestroy time: %lf ms\n",
           GET_DURING(t1, t0), GET_DURING(t2, t1), GET_DURING(t3, t2), GET_DURING(t4, t3), GET_DURING(t5, t4),
           GET_DURING(t6, t5));

    printf("total time: %lf ms\n", GET_DURING(t6, start_time));
    return 0;
}
