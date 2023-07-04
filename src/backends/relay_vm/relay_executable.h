/*!
 * @file relay_vm/relay_executable.h
 * @brief private struct and functions for relay executable.
 * @author YangBo MG21330067@smail.nju.edu.cn
 * @sa https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/executable.h
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_EXECUTABLE_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_EXECUTABLE_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <relay_vm/relay_instruction.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/trie.h>

/** @brief Relay Executable */
typedef struct TVM_RT_WASM_RelayExecutable_st *TVM_RT_WASM_RelayExecutable;

/** @brief Relay Executable definition */
struct TVM_RT_WASM_RelayExecutable_st {
    /*! @brief module handle */
    TVMModuleHandle module_handle;

    /*! @brief devices */
    DLDevice *devices;

    /*! @brief global string map to index */
    Trie *global_map;

    /*! @brief packed functions */
    TVMFunctionHandle *packed_functions;

    /*! @brief constant tensors */
    DLTensor *constant_tensors;
    /*! @brief the constant may not be immediate, use the late bound name to index. */
    char **late_bound_constant_names;
    /*! @brief constant tensors device index */
    size_t *constant_device_indices;

    /*! @brief relay VM functions */
    TVM_RT_WASM_RelayFunction *functions;

    size_t num_constant_tensors;
    size_t num_functions;
    size_t num_packed_functions;
    size_t num_devices;
    size_t host_device_index;
};

/**
 * @brief Load the relay executable from stream reader.
 * @param module_handle TVM relay executable library module. If NULL, use the system library.
 * @param reader The stream reader instance.
 * @param devices The physical devices.
 * @param num_dev The number of physical devices.
 * @param exe_ptr The pointer to receive TVM_RT_WASM_RelayExecutable.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayExecutableCreateFromReader(TVMModuleHandle module_handle, StreamReader *reader,
                                                const DLDevice *devices, uint32_t num_dev,
                                                TVM_RT_WASM_RelayExecutable *exe_ptr);

/*!
 * @brief Free the instance of TVM_RT_WASM_RelayExecutable.
 * @param exe The instance of TVM_RT_WASM_RelayExecutable.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayExecutableFree(TVM_RT_WASM_RelayExecutable exe);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_EXECUTABLE_H_INCLUDE_
