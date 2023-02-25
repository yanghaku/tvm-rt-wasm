#include "during.h"
#include "graph_utils.h"
#include "tvm_error_process.h"
#include <dlpack/dlpack.h>
#include <float.h>
#include <tvm/runtime/graph_executor_manager.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

#define INPUT_SHAPE_0 (1 * 14)
#define INPUT_SHAPE_1 (1 * 14)
#define OUTPUT_SHAPE_0 (1 * 14 * 1024)
#define OUTPUT_SHAPE_1 (1 * 1024)

extern const unsigned char graph_json[];

int main(int argc, char **argv) {
    SET_TIME(start_time)

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph.params>\n", __FILE__);
        return -1;
    }

    // static array storage
    static int64_t input_storage[INPUT_SHAPE_0 + INPUT_SHAPE_1];
    static float output_storage_0[OUTPUT_SHAPE_0];
    static float output_storage_1[OUTPUT_SHAPE_1];

    // local variables
    const DLDevice cpu = {kDLCPU, 0};
    const DLDataType float32 = {kDLFloat, 32, 1};
    const DLDataType int64 = {kDLInt, 64, 1};
    const int64_t input_shape_0[] = {1, 14};
    const int64_t input_shape_1[] = {1, 14};
    const int64_t out_shape_0[] = {1, 14, 1024};
    const int64_t out_shape_1[] = {1, 1024};
    const DLTensor inputs[] = {{
                                   .data = input_storage,
                                   .device = cpu,
                                   .ndim = 2,
                                   .dtype = int64,
                                   .shape = input_shape_0,
                                   .strides = NULL,
                                   .byte_offset = 0,
                               },
                               {
                                   .data = input_storage + INPUT_SHAPE_0,
                                   .device = cpu,
                                   .ndim = 2,
                                   .dtype = int64,
                                   .shape = input_shape_1,
                                   .strides = NULL,
                                   .byte_offset = 0,
                               }};
    const DLTensor outputs[] = {{
                                    .data = output_storage_0,
                                    .device = cpu,
                                    .ndim = 3,
                                    .dtype = float32,
                                    .shape = out_shape_0,
                                    .strides = NULL,
                                    .byte_offset = 0,
                                },
                                {
                                    .data = output_storage_1,
                                    .device = cpu,
                                    .ndim = 2,
                                    .dtype = float32,
                                    .shape = out_shape_1,
                                    .strides = NULL,
                                    .byte_offset = 0,
                                }};

    const char *input_names[] = {"input_ids", "attention_mask"};
    const int output_indexes[] = {0, 1};

    int status;
    GraphExecutorManager *graphManager = NULL;
    RUN(init_graph_with_syslib(argv[1], (const char *)graph_json, &graphManager));

    while (1) {
        // load input from file
        size_t len = fread(input_storage, sizeof(int64_t), INPUT_SHAPE_0 + INPUT_SHAPE_1, stdin);
        if (len != INPUT_SHAPE_0 + INPUT_SHAPE_1) {
            if (feof(stdin)) {
                break;
            }
            fprintf(stderr, INPUT_ERR_MSG);
            return -1;
        }

        RUN(run_graph(graphManager, inputs, input_names, 2, outputs, output_indexes, 2));

        float max_iter = -FLT_MAX;
        float min_iter = FLT_MAX;
        int32_t max_index = -1;
        int32_t min_index = -1;
        for (int32_t i = 0; i < OUTPUT_SHAPE_0; ++i) {
            if (output_storage_0[i] > max_iter) {
                max_iter = output_storage_0[i];
                max_index = i;
            }
            if (output_storage_0[i] < min_iter) {
                min_iter = output_storage_0[i];
                min_index = i;
            }
        }

        // 7.272131 -4.657999
        printf("The maximum position in output vector is: %d, with max-value %f.\n", max_index, max_iter);
        printf("The minimum position in output vector is: %d, with min-value %f.\n", min_index, min_iter);
    }

    RUN(graphManager->Release(&graphManager));

    SET_TIME(end_time)
    printf("\nTotal time: %lf ms\n", GET_DURING(end_time, start_time));
    return 0;
}
