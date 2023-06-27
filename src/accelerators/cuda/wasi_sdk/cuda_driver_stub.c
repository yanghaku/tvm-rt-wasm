/*!
 * \file cuda/wasi_sdk/cuda_driver_stub.c
 * \brief cuda driver error functions for wasm32-wasi target
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#if defined(__wasm32__) && defined(__wasi__) && (__wasm32__ == 1) && (__wasi__ == 1)

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CUresult cuGetErrorName(CUresult error, const char **p_str) {
    if (!p_str) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    switch (error) {
    case CUDA_SUCCESS:
        *p_str = "CUDA_SUCCESS";
        break;
    case CUDA_ERROR_INVALID_VALUE:
        *p_str = "CUDA_ERROR_INVALID_VALUE";
        break;
    case CUDA_ERROR_OUT_OF_MEMORY:
        *p_str = "CUDA_ERROR_OUT_OF_MEMORY";
        break;
    case CUDA_ERROR_NOT_INITIALIZED:
        *p_str = "CUDA_ERROR_NOT_INITIALIZED";
        break;
    case CUDA_ERROR_DEINITIALIZED:
        *p_str = "CUDA_ERROR_DEINITIALIZED";
        break;
    case CUDA_ERROR_PROFILER_DISABLED:
        *p_str = "CUDA_ERROR_PROFILER_DISABLED";
        break;
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        *p_str = "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
        break;
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        *p_str = "CUDA_ERROR_PROFILER_ALREADY_STARTED";
        break;
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        *p_str = "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
        break;
    case CUDA_ERROR_STUB_LIBRARY:
        *p_str = "CUDA_ERROR_STUB_LIBRARY";
        break;
    case CUDA_ERROR_DEVICE_UNAVAILABLE:
        *p_str = "CUDA_ERROR_DEVICE_UNAVAILABLE";
        break;
    case CUDA_ERROR_NO_DEVICE:
        *p_str = "CUDA_ERROR_NO_DEVICE";
        break;
    case CUDA_ERROR_INVALID_DEVICE:
        *p_str = "CUDA_ERROR_INVALID_DEVICE";
        break;
    case CUDA_ERROR_DEVICE_NOT_LICENSED:
        *p_str = "CUDA_ERROR_DEVICE_NOT_LICENSED";
        break;
    case CUDA_ERROR_INVALID_IMAGE:
        *p_str = "CUDA_ERROR_INVALID_IMAGE";
        break;
    case CUDA_ERROR_INVALID_CONTEXT:
        *p_str = "CUDA_ERROR_INVALID_CONTEXT";
        break;
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        *p_str = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
        break;
    case CUDA_ERROR_MAP_FAILED:
        *p_str = "CUDA_ERROR_MAP_FAILED";
        break;
    case CUDA_ERROR_UNMAP_FAILED:
        *p_str = "CUDA_ERROR_UNMAP_FAILED";
        break;
    case CUDA_ERROR_ARRAY_IS_MAPPED:
        *p_str = "CUDA_ERROR_ARRAY_IS_MAPPED";
        break;
    case CUDA_ERROR_ALREADY_MAPPED:
        *p_str = "CUDA_ERROR_ALREADY_MAPPED";
        break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
        *p_str = "CUDA_ERROR_NO_BINARY_FOR_GPU";
        break;
    case CUDA_ERROR_ALREADY_ACQUIRED:
        *p_str = "CUDA_ERROR_ALREADY_ACQUIRED";
        break;
    case CUDA_ERROR_NOT_MAPPED:
        *p_str = "CUDA_ERROR_NOT_MAPPED";
        break;
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        *p_str = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
        break;
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        *p_str = "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
        break;
    case CUDA_ERROR_ECC_UNCORRECTABLE:
        *p_str = "CUDA_ERROR_ECC_UNCORRECTABLE";
        break;
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
        *p_str = "CUDA_ERROR_UNSUPPORTED_LIMIT";
        break;
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        *p_str = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
        break;
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
        *p_str = "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
        break;
    case CUDA_ERROR_INVALID_PTX:
        *p_str = "CUDA_ERROR_INVALID_PTX";
        break;
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
        *p_str = "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT";
        break;
    case CUDA_ERROR_NVLINK_UNCORRECTABLE:
        *p_str = "CUDA_ERROR_NVLINK_UNCORRECTABLE";
        break;
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
        *p_str = "CUDA_ERROR_JIT_COMPILER_NOT_FOUND";
        break;
    case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
        *p_str = "CUDA_ERROR_UNSUPPORTED_PTX_VERSION";
        break;
    case CUDA_ERROR_JIT_COMPILATION_DISABLED:
        *p_str = "CUDA_ERROR_JIT_COMPILATION_DISABLED";
        break;
    case CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY:
        *p_str = "CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY";
        break;
    case CUDA_ERROR_INVALID_SOURCE:
        *p_str = "CUDA_ERROR_INVALID_SOURCE";
        break;
    case CUDA_ERROR_FILE_NOT_FOUND:
        *p_str = "CUDA_ERROR_FILE_NOT_FOUND";
        break;
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        *p_str = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
        break;
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        *p_str = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
        break;
    case CUDA_ERROR_OPERATING_SYSTEM:
        *p_str = "CUDA_ERROR_OPERATING_SYSTEM";
        break;
    case CUDA_ERROR_INVALID_HANDLE:
        *p_str = "CUDA_ERROR_INVALID_HANDLE";
        break;
    case CUDA_ERROR_ILLEGAL_STATE:
        *p_str = "CUDA_ERROR_ILLEGAL_STATE";
        break;
    case CUDA_ERROR_NOT_FOUND:
        *p_str = "CUDA_ERROR_NOT_FOUND";
        break;
    case CUDA_ERROR_NOT_READY:
        *p_str = "CUDA_ERROR_NOT_READY";
        break;
    case CUDA_ERROR_ILLEGAL_ADDRESS:
        *p_str = "CUDA_ERROR_ILLEGAL_ADDRESS";
        break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        *p_str = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
        break;
    case CUDA_ERROR_LAUNCH_TIMEOUT:
        *p_str = "CUDA_ERROR_LAUNCH_TIMEOUT";
        break;
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        *p_str = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
        break;
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        *p_str = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
        break;
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        *p_str = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
        break;
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        *p_str = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
        break;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        *p_str = "CUDA_ERROR_CONTEXT_IS_DESTROYED";
        break;
    case CUDA_ERROR_ASSERT:
        *p_str = "CUDA_ERROR_ASSERT";
        break;
    case CUDA_ERROR_TOO_MANY_PEERS:
        *p_str = "CUDA_ERROR_TOO_MANY_PEERS";
        break;
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        *p_str = "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
        break;
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        *p_str = "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
        break;
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
        *p_str = "CUDA_ERROR_HARDWARE_STACK_ERROR";
        break;
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
        *p_str = "CUDA_ERROR_ILLEGAL_INSTRUCTION";
        break;
    case CUDA_ERROR_MISALIGNED_ADDRESS:
        *p_str = "CUDA_ERROR_MISALIGNED_ADDRESS";
        break;
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
        *p_str = "CUDA_ERROR_INVALID_ADDRESS_SPACE";
        break;
    case CUDA_ERROR_INVALID_PC:
        *p_str = "CUDA_ERROR_INVALID_PC";
        break;
    case CUDA_ERROR_LAUNCH_FAILED:
        *p_str = "CUDA_ERROR_LAUNCH_FAILED";
        break;
    case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
        *p_str = "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE";
        break;
    case CUDA_ERROR_NOT_PERMITTED:
        *p_str = "CUDA_ERROR_NOT_PERMITTED";
        break;
    case CUDA_ERROR_NOT_SUPPORTED:
        *p_str = "CUDA_ERROR_NOT_SUPPORTED";
        break;
    case CUDA_ERROR_SYSTEM_NOT_READY:
        *p_str = "CUDA_ERROR_SYSTEM_NOT_READY";
        break;
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
        *p_str = "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH";
        break;
    case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
        *p_str = "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE";
        break;
    case CUDA_ERROR_MPS_CONNECTION_FAILED:
        *p_str = "CUDA_ERROR_MPS_CONNECTION_FAILED";
        break;
    case CUDA_ERROR_MPS_RPC_FAILURE:
        *p_str = "CUDA_ERROR_MPS_RPC_FAILURE";
        break;
    case CUDA_ERROR_MPS_SERVER_NOT_READY:
        *p_str = "CUDA_ERROR_MPS_SERVER_NOT_READY";
        break;
    case CUDA_ERROR_MPS_MAX_CLIENTS_REACHED:
        *p_str = "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED";
        break;
    case CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED:
        *p_str = "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED";
        break;
    case CUDA_ERROR_MPS_CLIENT_TERMINATED:
        *p_str = "CUDA_ERROR_MPS_CLIENT_TERMINATED";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_INVALIDATED";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_MERGE:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_MERGE";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_UNMATCHED";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_UNJOINED";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_ISOLATION";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_IMPLICIT";
        break;
    case CUDA_ERROR_CAPTURED_EVENT:
        *p_str = "CUDA_ERROR_CAPTURED_EVENT";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
        *p_str = "CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD";
        break;
    case CUDA_ERROR_TIMEOUT:
        *p_str = "CUDA_ERROR_TIMEOUT";
        break;
    case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
        *p_str = "CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE";
        break;
    case CUDA_ERROR_EXTERNAL_DEVICE:
        *p_str = "CUDA_ERROR_EXTERNAL_DEVICE";
        break;
    case CUDA_ERROR_INVALID_CLUSTER_SIZE:
        *p_str = "CUDA_ERROR_INVALID_CLUSTER_SIZE";
        break;
    case CUDA_ERROR_UNKNOWN:
        *p_str = "CUDA_ERROR_UNKNOWN";
        break;
    default:
        *p_str = (const char *)0;
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **p_str) {
    if (!p_str) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (error) {
    case CUDA_SUCCESS:
        *p_str = "no error";
        break;
    case CUDA_ERROR_INVALID_VALUE:
        *p_str = "invalid argument";
        break;
    case CUDA_ERROR_OUT_OF_MEMORY:
        *p_str = "out of memory";
        break;
    case CUDA_ERROR_NOT_INITIALIZED:
        *p_str = "initialization error";
        break;
    case CUDA_ERROR_DEINITIALIZED:
        *p_str = "driver shutting down";
        break;
    case CUDA_ERROR_PROFILER_DISABLED:
        *p_str = "profiler disabled while using external profiling tool";
        break;
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        *p_str = "profiler not initialized: call cudaProfilerInitialize()";
        break;
    case CUDA_ERROR_PROFILER_ALREADY_STARTED:
        *p_str = "profiler already started";
        break;
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        *p_str = "profiler already stopped";
        break;
    case CUDA_ERROR_STUB_LIBRARY:
        *p_str = "CUDA driver is a stub library";
        break;
    case CUDA_ERROR_DEVICE_UNAVAILABLE:
        *p_str = "CUDA-capable device(s) is/are busy or unavailable";
        break;
    case CUDA_ERROR_NO_DEVICE:
        *p_str = "no CUDA-capable device is detected";
        break;
    case CUDA_ERROR_INVALID_DEVICE:
        *p_str = "invalid device ordinal";
        break;
    case CUDA_ERROR_DEVICE_NOT_LICENSED:
        *p_str = "device doesn't have valid Grid license";
        break;
    case CUDA_ERROR_INVALID_IMAGE:
        *p_str = "device kernel image is invalid";
        break;
    case CUDA_ERROR_INVALID_CONTEXT:
        *p_str = "invalid device context";
        break;
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        *p_str = "context already current";
        break;
    case CUDA_ERROR_MAP_FAILED:
        *p_str = "mapping of buffer object failed";
        break;
    case CUDA_ERROR_UNMAP_FAILED:
        *p_str = "unmapping of buffer object failed";
        break;
    case CUDA_ERROR_ARRAY_IS_MAPPED:
        *p_str = "array is mapped";
        break;
    case CUDA_ERROR_ALREADY_MAPPED:
        *p_str = "resource already mapped";
        break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
        *p_str = "no kernel image is available for execution on the device";
        break;
    case CUDA_ERROR_ALREADY_ACQUIRED:
        *p_str = "resource already acquired";
        break;
    case CUDA_ERROR_NOT_MAPPED:
        *p_str = "resource not mapped";
        break;
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        *p_str = "resource not mapped as array";
        break;
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        *p_str = "resource not mapped as pointer";
        break;
    case CUDA_ERROR_ECC_UNCORRECTABLE:
        *p_str = "uncorrectable ECC error encountered";
        break;
    case CUDA_ERROR_UNSUPPORTED_LIMIT:
        *p_str = "limit is not supported on this architecture";
        break;
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        *p_str = "exclusive-thread device already in use by a different thread";
        break;
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
        *p_str = "peer access is not supported between these two devices";
        break;
    case CUDA_ERROR_INVALID_PTX:
        *p_str = "a PTX JIT compilation failed";
        break;
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
        *p_str = "invalid OpenGL or DirectX context";
        break;
    case CUDA_ERROR_NVLINK_UNCORRECTABLE:
        *p_str = "uncorrectable NVLink error detected during the execution";
        break;
    case CUDA_ERROR_JIT_COMPILER_NOT_FOUND:
        *p_str = "PTX JIT compiler library not found";
        break;
    case CUDA_ERROR_UNSUPPORTED_PTX_VERSION:
        *p_str = "the provided PTX was compiled with an unsupported toolchain.";
        break;
    case CUDA_ERROR_JIT_COMPILATION_DISABLED:
        *p_str = "PTX JIT compilation was disabled";
        break;
    case CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY:
        *p_str = "the provided execution affinity is not supported";
        break;
    case CUDA_ERROR_INVALID_SOURCE:
        *p_str = "device kernel image is invalid";
        break;
    case CUDA_ERROR_FILE_NOT_FOUND:
        *p_str = "file not found";
        break;
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        *p_str = "shared object symbol not found";
        break;
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        *p_str = "shared object initialization failed";
        break;
    case CUDA_ERROR_OPERATING_SYSTEM:
        *p_str = "OS call failed or operation not supported on this OS";
        break;
    case CUDA_ERROR_INVALID_HANDLE:
        *p_str = "invalid resource handle";
        break;
    case CUDA_ERROR_ILLEGAL_STATE:
        *p_str = "the operation cannot be performed in the present state";
        break;
    case CUDA_ERROR_NOT_FOUND:
        *p_str = "named symbol not found";
        break;
    case CUDA_ERROR_NOT_READY:
        *p_str = "device not ready";
        break;
    case CUDA_ERROR_ILLEGAL_ADDRESS:
        *p_str = "an illegal memory access was encountered";
        break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        *p_str = "too many resources requested for launch";
        break;
    case CUDA_ERROR_LAUNCH_TIMEOUT:
        *p_str = "the launch timed out and was terminated";
        break;
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        *p_str = "launch uses incompatible texturing mode";
        break;
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        *p_str = "peer access is already enabled";
        break;
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        *p_str = "peer access has not been enabled";
        break;
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        *p_str = "cannot set while device is active in this process";
        break;
    case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        *p_str = "context is destroyed";
        break;
    case CUDA_ERROR_ASSERT:
        *p_str = "device-side assert triggered";
        break;
    case CUDA_ERROR_TOO_MANY_PEERS:
        *p_str = "peer mapping resources exhausted";
        break;
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        *p_str = "part or all of the requested memory range is already mapped";
        break;
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        *p_str = "pointer does not correspond to a registered memory region";
        break;
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
        *p_str = "hardware stack error";
        break;
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
        *p_str = "an illegal instruction was encountered";
        break;
    case CUDA_ERROR_MISALIGNED_ADDRESS:
        *p_str = "misaligned address";
        break;
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
        *p_str = "operation not supported on global/shared address space";
        break;
    case CUDA_ERROR_INVALID_PC:
        *p_str = "invalid program counter";
        break;
    case CUDA_ERROR_LAUNCH_FAILED:
        *p_str = "unspecified launch failure";
        break;
    case CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE:
        *p_str = "too many blocks in cooperative launch";
        break;
    case CUDA_ERROR_NOT_PERMITTED:
        *p_str = "operation not permitted";
        break;
    case CUDA_ERROR_NOT_SUPPORTED:
        *p_str = "operation not supported";
        break;
    case CUDA_ERROR_SYSTEM_NOT_READY:
        *p_str = "system not yet initialized";
        break;
    case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
        *p_str = "system has unsupported display driver / cuda driver combination";
        break;
    case CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE:
        *p_str = "forward compatibility was attempted on non supported HW";
        break;
    case CUDA_ERROR_MPS_CONNECTION_FAILED:
        *p_str = "MPS client failed to connect to the MPS control daemon or the MPS server";
        break;
    case CUDA_ERROR_MPS_RPC_FAILURE:
        *p_str = "the remote procedural call between the MPS server and the MPS client failed";
        break;
    case CUDA_ERROR_MPS_SERVER_NOT_READY:
        *p_str = "MPS server is not ready to accept new MPS client requests";
        break;
    case CUDA_ERROR_MPS_MAX_CLIENTS_REACHED:
        *p_str = "the hardware resources required to create MPS client have been exhausted";
        break;
    case CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED:
        *p_str =
            "the hardware resources required to support device connections have been exhausted";
        break;
    case CUDA_ERROR_MPS_CLIENT_TERMINATED:
        *p_str = "the MPS client has been terminated by the server";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED:
        *p_str = "operation not permitted when stream is capturing";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_INVALIDATED:
        *p_str = "operation failed due to a previous error during capture";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_MERGE:
        *p_str = "operation would result in a merge of separate capture sequences";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNMATCHED:
        *p_str = "capture was not ended in the same stream as it began";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_UNJOINED:
        *p_str = "capturing stream has unjoined work";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_ISOLATION:
        *p_str = "dependency created on uncaptured work in another stream";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_IMPLICIT:
        *p_str = "operation would make the legacy stream depend on a capturing blocking stream";
        break;
    case CUDA_ERROR_CAPTURED_EVENT:
        *p_str = "operation not permitted on an event last recorded in a capturing stream";
        break;
    case CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD:
        *p_str = "attempt to terminate a thread-local capture sequence from another thread";
        break;
    case CUDA_ERROR_TIMEOUT:
        *p_str = "wait operation timed out";
        break;
    case CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE:
        *p_str = "the graph update was not performed because it included changes which violated "
                 "constraints specific to instantiated graph update";
        break;
    case CUDA_ERROR_EXTERNAL_DEVICE:
        *p_str = "an async error has occured in external entity outside of CUDA";
        break;
    case CUDA_ERROR_INVALID_CLUSTER_SIZE:
        *p_str = "a kernel launch error has occurred due to cluster misconfiguration";
        break;
    case CUDA_ERROR_UNKNOWN:
        *p_str = "unknown error";
        break;
    default:
        *p_str = (const char *)0;
        return CUDA_ERROR_INVALID_VALUE;
    }

    return CUDA_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // wasm32-wasi
