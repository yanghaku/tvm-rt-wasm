#include "during.h"
#include "graph_utils.h"
#include "tvm_error_process.h"
#include <dlpack/dlpack.h>
#include <float.h>
#include <tvm/runtime/graph_executor_manager.h>

#define OUTPUT_LEN 1000
#define INPUT_SHAPE (1 * 3 * 224 * 224)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

extern const unsigned char graph_json[];

int main(int argc, char **argv) {
    SET_TIME(start_time)

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph.params>\n", __FILE__);
        return -1;
    }

    // static array storage
    static float input_storage[INPUT_SHAPE];
    static float output_storage[OUTPUT_LEN];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    int64_t input_shape[4] = {1, 3, 224, 224};
    int64_t out_shape[2] = {1, 1000};
    DLTensor input = {
        .data = input_storage,
        .device = cpu,
        .ndim = 4,
        .dtype = float32,
        .shape = input_shape,
        .strides = NULL,
        .byte_offset = 0,
    };
    DLTensor output = {
        .data = output_storage,
        .device = cpu,
        .ndim = 2,
        .dtype = float32,
        .shape = out_shape,
        .strides = NULL,
        .byte_offset = 0,
    };

    int status;
    GraphExecutorManager *graphManager = NULL;
    RUN(init_graph_with_syslib(argv[1], (const char *)graph_json, &graphManager));

    while (1) {
        // load input from file
        size_t len = fread(input_storage, 4, INPUT_SHAPE, stdin);
        if (len != INPUT_SHAPE) {
            if (feof(stdin)) {
                break;
            }
            fprintf(stderr, INPUT_ERR_MSG);
            return -1;
        }

        RUN(run_graph(graphManager, &input, &output, "input", 0));

        float max_iter = -FLT_MAX;
        int32_t max_index = -1;
        for (int i = 0; i < OUTPUT_LEN; ++i) {
            if (output_storage[i] > max_iter) {
                max_iter = output_storage[i];
                max_index = i;
            }
        }
        printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);
    }

    RUN(graphManager->Release(&graphManager));

    SET_TIME(end_time)
    printf("\nTotal time: %lf ms\n", GET_DURING(end_time, start_time));
    return 0;
}
