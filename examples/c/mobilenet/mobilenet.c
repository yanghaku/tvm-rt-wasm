#include "graph_utils.h"
#include <float.h>

#define OUTPUT_LEN 1000
#define INPUT_SHAPE (1 * 3 * 224 * 224)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

#ifndef DSO_TEST
extern const unsigned char graph_json[];
#else
#define GRAPH_JSON_MAX_LEN 80000
static char graph_json[GRAPH_JSON_MAX_LEN];
#endif // !DSO_TEST

int main(int argc, char **argv) {
    SET_TIME(start_time)

#ifdef DSO_TEST
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <graph.so> <graph.params> <graph.json>\n", argv[0]);
        return -1;
    }
    FILE *j = fopen(argv[3], "r");
    if (!j) {
        fprintf(stderr, "Cannot open graph json file `%s`\n", argv[3]);
        return -2;
    }
    size_t graph_json_len = fread(graph_json, 1, GRAPH_JSON_MAX_LEN, j);
    fclose(j);
    graph_json[graph_json_len] = '\0';
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <graph.params>\n", __FILE__);
        return -1;
    }
#endif // DSO_TEST

    // static array storage
    static float input_storage[INPUT_SHAPE];
    static float output_storage[OUTPUT_LEN];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    int64_t input_shape[4] = {1, 3, 224, 224};
    int64_t out_shape[2] = {1, 1000};
    const DLTensor inputs[] = {{
        .data = input_storage,
        .device = cpu,
        .ndim = 4,
        .dtype = float32,
        .shape = input_shape,
        .strides = NULL,
        .byte_offset = 0,
    }};
    DLTensor outputs[] = {{
        .data = output_storage,
        .device = cpu,
        .ndim = 2,
        .dtype = float32,
        .shape = out_shape,
        .strides = NULL,
        .byte_offset = 0,
    }};
    const char *input_names[] = {"input"};
    const int output_indexes[] = {0};

    int status;
    GraphHandle graph_handle = NULL;
#ifdef DSO_TEST
    RUN(init_graph_with_dso_lib(argv[1], argv[2], (const char *)graph_json, &graph_handle));
#else
    RUN(init_graph_with_syslib(argv[1], (const char *)graph_json, &graph_handle));
#endif // DSO_TEST

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

        RUN(run_graph(graph_handle, inputs, input_names, 1, outputs, output_indexes, 1));

        float max_iter = -FLT_MAX;
        int32_t max_index = -1;
        for (int i = 0; i < OUTPUT_LEN; ++i) {
            if (output_storage[i] > max_iter) {
                max_iter = output_storage[i];
                max_index = i;
            }
        }
        printf("The maximum position in output vector is: %d, with max-value %f. END\n", max_index,
               max_iter);
        fflush(stdout);
    }

    RUN(delete_graph(graph_handle));

    SET_TIME(end_time)
    printf("\nTotal time: %lf ms\n", GET_DURING(end_time, start_time));
    return 0;
}
