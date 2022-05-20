/*!
 * \file src/runtime/device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/device/cpu_memory.h>
#include <tvm/runtime/device/cuda_device_api.h>
#include <tvm/runtime/utils/tensor_helper.h>

#if USE_CUDA // USE_CUDA = 1

/*! \brief the cuda Device API will be a single static instance */
static CUDADeviceAPI cudaDeviceApi;

static int TVM_RT_WASM_CUDA_SetDevice(int dev_id) {
    CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
    return 0;
}

static void *TVM_RT_WASM_CUDA_AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL_NULL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
    return res;
}

static void *TVM_RT_WASM_CUDA_AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype,
                                                  const char *mem_scope) {
    SET_ERROR_RETURN(-1, "%s is not supported yet\n", __FUNCTION__);
}

static int TVM_RT_WASM_CUDA_FreeDataSpace(int dev_id, void *ptr) {
    CUDA_DRIVER_CALL(cuMemFree((CUdeviceptr)ptr));
    return 0;
}

static int TVM_RT_WASM_CUDA_CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
    uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(from);
    uint64_t byte_check = TVM_RT_WASM_DLTensor_GetDataBytes(to);
    if (unlikely(bytes != byte_check)) {
        SET_ERROR_RETURN(-1, "Error: data copy size is diff, from=%lld and to=%lld\n", bytes, byte_check);
    }
    if (from->device.device_type == kDLCPU || from->device.device_type == kDLCUDAHost) {
        if (to->device.device_type == kDLCUDAHost || to->device.device_type == kDLCPU) {
            memcpy(to->data, from->data, bytes);
        } else if (to->device.device_type == kDLCUDA) {
            CUDA_DRIVER_CALL(cuMemcpyHtoD((CUdeviceptr)to->data, from->data, bytes));
        } else {
            SET_ERROR_RETURN(-1, "Error: unsupported data copy!\n");
        }
    } else if (from->device.device_type == kDLCUDA) {
        if (to->device.device_type == kDLCPU || to->device.device_type == kDLCUDAHost) {
            CUDA_DRIVER_CALL(cuMemcpyDtoH(to->data, (CUdeviceptr)from->data, bytes));
        } else if (to->device.device_type == kDLCUDA) {
            CUDA_DRIVER_CALL(cuMemcpyDtoD((CUdeviceptr)to->data, (CUdeviceptr)from->data, bytes));
        } else {
            SET_ERROR_RETURN(-1, "Error: unsupported data copy!\n");
        }
    }
    return 0;
}

static TVMStreamHandle TVM_RT_WASM_CUDA_CreateStream(int dev_id) {
    CUstream out;
    CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
    CUDA_DRIVER_CALL_NULL(cuStreamCreate(&out, CU_STREAM_DEFAULT));
    return out;
}

static int TVM_RT_WASM_CUDA_FreeStream(int dev_id, TVMStreamHandle stream) {
    CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
    CUDA_DRIVER_CALL(cuStreamDestroy(stream));
    return 0;
}

static int TVM_RT_WASM_CUDA_StreamSync(int dev_id, TVMStreamHandle stream) {
    CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
    CUDA_DRIVER_CALL(cuStreamSynchronize(stream));
    return 0;
}

static int TVM_RT_WASM_CUDA_SetStream(int dev_id, TVMStreamHandle stream) {
    cudaDeviceApi.stream = stream;
    return 0;
}

static TVMStreamHandle TVM_RT_WASM_CUDA_GetStream() { return cudaDeviceApi.stream; }

static void TVM_RT_WASM_CUDA_SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {}

static void *TVM_RT_WASM_CUDA_AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *res = NULL;

#ifdef CUDA_10_ONLY
    CUDA_DRIVER_CALL_NULL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
#else
    CUDA_DRIVER_CALL_NULL(
        cuMemAllocFromPoolAsync((CUdeviceptr *)&res, nbytes, cudaDeviceApi.mem_pool, cudaDeviceApi.stream));
#endif // CUDA_10_ONLY

    return res;
}

static int TVM_RT_WASM_CUDA_FreeWorkspace(int dev_id, void *ptr) {
#ifdef CUDA_10_ONLY
    CUDA_DRIVER_CALL(cuMemFree((CUdeviceptr)ptr));
#else
    CUDA_DRIVER_CALL(cuMemFreeAsync((CUdeviceptr)ptr, cudaDeviceApi.stream));
#endif // CUDA_10_ONLY
    return 0;
}

static int TVM_RT_WASM_CUDA_Release(DeviceAPI *d) {
    if (d != (DeviceAPI *)&cudaDeviceApi)
        return -1;
    for (uint32_t i = 0; i < cudaDeviceApi.num_device; ++i) {
        CUDA_DRIVER_CALL(cuCtxDestroy(cudaDeviceApi.contexts[i]));
    }
    TVM_RT_WASM_HeapMemoryFree(cudaDeviceApi.contexts);
    return 0;
}

#endif

/*!
 * \brief create a instance of cuda device api
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int TVM_RT_WASM_CUDADeviceAPICreate(CUDADeviceAPI **out) {

#if USE_CUDA // USE_CUDA = 1

    *out = &cudaDeviceApi;

    cudaDeviceApi.SetDevice = TVM_RT_WASM_CUDA_SetDevice;
    cudaDeviceApi.AllocDataSpace = TVM_RT_WASM_CUDA_AllocDataSpace;
    cudaDeviceApi.AllocDataSpaceScope = TVM_RT_WASM_CUDA_AllocDataSpaceScope;
    cudaDeviceApi.FreeDataSpace = TVM_RT_WASM_CUDA_FreeDataSpace;
    cudaDeviceApi.CopyDataFromTo = TVM_RT_WASM_CUDA_CopyDataFromTo;
    cudaDeviceApi.CreateStream = TVM_RT_WASM_CUDA_CreateStream;
    cudaDeviceApi.FreeStream = TVM_RT_WASM_CUDA_FreeStream;
    cudaDeviceApi.StreamSync = TVM_RT_WASM_CUDA_StreamSync;
    cudaDeviceApi.SetStream = TVM_RT_WASM_CUDA_SetStream;
    cudaDeviceApi.GetStream = TVM_RT_WASM_CUDA_GetStream;
    cudaDeviceApi.SyncStreamFromTo = TVM_RT_WASM_CUDA_SyncStreamFromTo;
    cudaDeviceApi.AllocWorkspace = TVM_RT_WASM_CUDA_AllocWorkspace;
    cudaDeviceApi.FreeWorkspace = TVM_RT_WASM_CUDA_FreeWorkspace;
    cudaDeviceApi.Release = TVM_RT_WASM_CUDA_Release;

    SET_TIME(t0)
    CUDA_DRIVER_CALL(cuInit(0));
    SET_TIME(t1)

    int num_device = 0;
    CUDA_DRIVER_CALL(cuDeviceGetCount(&num_device));
    cudaDeviceApi.num_device = num_device;

#ifndef CUDA_10_ONLY
    CUDA_DRIVER_CALL(cuDeviceGetDefaultMemPool(&cudaDeviceApi.mem_pool, 0));
#endif // CUDA_10_ONLY

    SET_TIME(t2)
    cudaDeviceApi.contexts = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUcontext) * num_device);
    for (int i = 0; i < num_device; ++i) {
        CUDA_DRIVER_CALL(cuCtxCreate(cudaDeviceApi.contexts + i, 0, i));
    }
    SET_TIME(t3)
    cudaDeviceApi.stream = NULL;

    DURING_PRINT(t1, t0, "cuInit time");
    DURING_PRINT(t3, t2, "CUcontext create time");
    return 0;

#else
    CUDA_NOT_SUPPORTED();
#endif
}
