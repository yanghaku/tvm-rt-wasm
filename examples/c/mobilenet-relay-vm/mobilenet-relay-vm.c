#include "dlpack/dlpack.h"
#include "during.h"
#include "relay_vm.h"
#include "tvm_error_process.h"
#include <float.h>

#define OUTPUT_LEN 1000
#define INPUT_SHAPE (1 * 3 * 224 * 224)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define INPUT_ERR_MSG "Unexpected EOF, input should be shape with " TOSTRING(INPUT_SHAPE) "\n"

int main(int argc, char **argv) {
    SET_TIME(start_time)

    int status;
#ifdef DSO_TEST
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <relay executable bytecode> <relay executable library>\n",
                argv[0]);
        return -1;
    }
    TVMModuleHandle module = NULL;
    RUN(TVMModLoadFromFile(argv[2], "so", &module));
#else
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <relay executable bytecode>\n", argv[0]);
        return -1;
    }
    TVMModuleHandle module = NULL;
#endif // DSO_TEST

    // static array storage
    static float input_storage[INPUT_SHAPE];
    static float output_storage[OUTPUT_LEN];

    // local variables
    DLDevice cpu = {kDLCPU, 0};
    DLDataType float32 = {kDLFloat, 32, 1};
    int64_t input_shape[4] = {1, 3, 224, 224};
    int64_t out_shape[2] = {1, 1000};
    const DLTensor input = {
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

#if EXAMPLE_USE_CUDA
    DLDevice exec_device = {kDLCUDA, 0};
#elif EXAMPLE_USE_WEBGPU
    DLDevice exec_device = {kDLWebGPU, 0};
#else
    DLDevice exec_device = cpu;
#endif

    SET_TIME(create_start)
    TVM_RT_WASM_RelayVirtualMachine vm =
        TVM_RT_WASM_RelayVirtualMachineCreateFromFile(module, argv[1], &exec_device, 1);
    if (!vm) {
        fprintf(stderr, "Create relay vm fail: %s\n", TVMGetLastError());
        return -1;
    }
    SET_TIME(create_end)
    printf("Create relay vm time: %lf ms\n", GET_DURING(create_end, create_start));

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

        SET_TIME(t0) // set input start

        RUN(TVM_RT_WASM_RelayVirtualMachineSetInput(vm, NULL, 0, &input));

        SET_TIME(t1) // set input end, run start

        RUN(TVM_RT_WASM_RelayVirtualMachineRun(vm, NULL));

        SET_TIME(t2) // run end, get output start

        RUN(TVM_RT_WASM_RelayVirtualMachineGetOutput(vm, NULL, 0, &output));

        SET_TIME(t3) // get output end
        printf("Set input time: %lf ms\nRun time: %lf ms\nGet output time: %lf ms\n",
               GET_DURING(t1, t0), GET_DURING(t2, t1), GET_DURING(t3, t2));

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

    RUN(TVM_RT_WASM_RelayVirtualMachineFree(vm));

    SET_TIME(end_time)
    printf("\nTotal time: %lf ms\n", GET_DURING(end_time, start_time));

#ifdef DSO_TEST
    RUN(TVMModFree(module));
#endif // DSO_TEST
    return 0;
}
