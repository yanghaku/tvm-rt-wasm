/*!
 * \file aot/aot_executor.h
 * \brief aot_executor struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_BACKENDS_AOT_AOT_EXECUTOR_INNER_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_AOT_AOT_EXECUTOR_INNER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <aot_executor.h>
#include <module/module.h>
#include <tvm/runtime/metadata_types.h>
#include <utils/common.h>

struct TVM_RT_WASM_AotExecutor_st {
    //    TVMModuleHandle module_handle;
    const struct TVMMetadata *metadata;
    DLDevice *devices;
    //    uint32_t num_devices;

    DLTensor *tensors;

    PackedFunction *main_func;
    TVMValue *tvm_args_value;
    int *tvm_args_type;
    int tvm_args_size;
};

#define CHECK_AotExecutor(g) CHECK_INPUT_POINTER(g, -2, "AotExecutor")

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_AOT_AOT_EXECUTOR_INNER_H_INCLUDE_
