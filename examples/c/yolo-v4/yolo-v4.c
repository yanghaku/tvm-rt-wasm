#include "during.h"
#include "tvm_error_process.h"
#include <dlpack/dlpack.h>
#include <tvm/runtime/graph_executor_manager.h>

#define OUTPUT_LEN_0 (1 * 425 * 13 * 13)
#define OUTPUT_LEN_1 (10)
#define OUTPUT_LEN_2 (7)
#define GRAPH_PARAMS_SIZE 195 * 1024 * 1024

#define INPUT_SHAPE (3 * 416 * 416)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

static int64_t input_shape[4] = {1, 3, 416, 416};
static int64_t out_shape_0[] = {1, 425, 13, 13};
static int64_t out_shape_1[] = {10};
static int64_t out_shape_2[] = {7};

static char graph_params[GRAPH_PARAMS_SIZE];
extern const unsigned char graph_json[];

int main(int argc, char **argv) {
    SET_TIME(start_time)

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph.params>\n", __FILE__);
        return -1;
    }

    // static array storage
    static float input_storage[INPUT_SHAPE];
    static float output_storage_0[OUTPUT_LEN_0];
    static float output_storage_1[OUTPUT_LEN_1];
    static int32_t output_storage_2[OUTPUT_LEN_2];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    DLTensor input, output[3];

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
    FILE *p = fopen(argv[1], "rb");
    if (p == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return -1;
    }
    size_t param_len = fread(graph_params, 1, GRAPH_PARAMS_SIZE, p);
    fclose(p);

    RUN(graphManager->LoadParams(graphManager, graph_params, param_len));

    SET_TIME(t2) // load graph end, set input start

    // load input from file
    size_t len = fread(input_storage, 4, INPUT_SHAPE, stdin);
    if (len != INPUT_SHAPE) {
        fprintf(stderr, INPUT_ERR_MSG);
        return -1;
    }
    input.data = input_storage;
    input.device = cpu;
    input.ndim = 4;
    input.dtype = float32;
    input.shape = input_shape;
    input.strides = NULL;
    input.byte_offset = 0;

    RUN(graphManager->SetInputByName(graphManager, "data", &input));

    SET_TIME(t3) // set input end, run graph start

    RUN(graphManager->Run(graphManager));

    SET_TIME(t4) // run end, get output start

    for (int i = 0; i < 3; ++i) {
        output[i].device = cpu;
        output[i].dtype = float32;
        output[i].strides = NULL;
        output[i].byte_offset = 0;
    }

    output[0].data = output_storage_0;
    output[0].ndim = 4;
    output[0].shape = out_shape_0;

    output[1].data = output_storage_1;
    output[1].ndim = 1;
    output[1].shape = out_shape_1;

    output[2].data = output_storage_2;
    output[2].ndim = 1;
    output[2].shape = out_shape_2;

    for (int i = 0; i < 3; ++i) {
        RUN(graphManager->GetOutput(graphManager, i, &output[i]));
    }

    SET_TIME(t5) // get output end, destroy start

    RUN(graphManager->Release(&graphManager));

    SET_TIME(t6)

    printf("create time: %lf ms\nload_params time: %lf ms\nset_input time: %lf\nrun time: %lf ms\nget_output time: %lf "
           "ms\ndestroy time: %lf ms\n",
           GET_DURING(t1, t0), GET_DURING(t2, t1), GET_DURING(t3, t2), GET_DURING(t4, t3), GET_DURING(t5, t4),
           GET_DURING(t6, t5));

    printf("total time: %lf ms\n", GET_DURING(t6, start_time));

    printf("output_1 = ");
    for (int i = 0; i < OUTPUT_LEN_1; ++i) {
        printf("%.10f, ", output_storage_1[i]);
    }
    printf("\noutput_2 = ");
    for (int i = 0; i < OUTPUT_LEN_2; ++i) {
        printf("%d, ", output_storage_2[i]);
    }
    printf("\n");
    return 0;
}
