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

static void TVM_RT_WASM_CUDA_SetDevice(int dev_id) {
    CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
}

static void *TVM_RT_WASM_CUDA_AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
    return res;
}

static void *TVM_RT_WASM_CUDA_AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype,
                                                  const char *mem_scope) {
    fprintf(stderr, "%s is not supported yet\n", __FUNCTION__);
    exit(-1);
}

static void TVM_RT_WASM_CUDA_FreeDataSpace(int dev_id, void *ptr) { CUDA_DRIVER_CALL(cuMemFree((CUdeviceptr)ptr)); }

static void TVM_RT_WASM_CUDA_CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
    uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(from);
    uint64_t byte_check = TVM_RT_WASM_DLTensor_GetDataBytes(to);
    if (unlikely(bytes != byte_check)) {
        fprintf(stderr, "Error: data copy size is diff, from=%lld and to=%lld\n", bytes, byte_check);
        exit(-1);
    }
    if (from->device.device_type == kDLCPU || from->device.device_type == kDLCUDAHost) {
        if (to->device.device_type == kDLCUDAHost || to->device.device_type == kDLCPU) {
            memcpy(to->data, from->data, bytes);
        } else if (to->device.device_type == kDLCUDA) {
            CUDA_DRIVER_CALL(cuMemcpyHtoD((CUdeviceptr)to->data, from->data, bytes));
        } else {
            fprintf(stderr, "Error: unsupported data copy!\n");
            exit(-1);
        }
    } else if (from->device.device_type == kDLCUDA) {
        if (to->device.device_type == kDLCPU || to->device.device_type == kDLCUDAHost) {
            CUDA_DRIVER_CALL(cuMemcpyDtoH(to->data, (CUdeviceptr)from->data, bytes));
        } else if (to->device.device_type == kDLCUDA) {
            CUDA_DRIVER_CALL(cuMemcpyDtoD((CUdeviceptr)to->data, (CUdeviceptr)from->data, bytes));
        } else {
            fprintf(stderr, "Error: unsupported data copy!\n");
            exit(-1);
        }
    }
}

static TVMStreamHandle TVM_RT_WASM_CUDA_CreateStream(int dev_id) {
    CUstream out;
    CUDA_DRIVER_CALL(cuStreamCreate(&out, CU_STREAM_DEFAULT));
    return out;
}

static void TVM_RT_WASM_CUDA_FreeStream(int dev_id, TVMStreamHandle stream) {
    CUDA_DRIVER_CALL(cuStreamDestroy(stream));
}

static void TVM_RT_WASM_CUDA_StreamSync(int dev_id, TVMStreamHandle stream) {
    CUDA_DRIVER_CALL(cuStreamSynchronize(stream));
}

static void TVM_RT_WASM_CUDA_SetStream(int dev_id, TVMStreamHandle stream) { cudaDeviceApi.stream = stream; }

static TVMStreamHandle TVM_RT_WASM_CUDA_GetStream() { return cudaDeviceApi.stream; }

static void TVM_RT_WASM_CUDA_SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {}

static void *TVM_RT_WASM_CUDA_AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL(
        cuMemAllocFromPoolAsync((CUdeviceptr *)&res, nbytes, cudaDeviceApi.mem_pool, cudaDeviceApi.stream));
    return res;
}

static void TVM_RT_WASM_CUDA_FreeWorkspace(int dev_id, void *ptr) {
    CUDA_DRIVER_CALL(cuMemFreeAsync((CUdeviceptr)ptr, cudaDeviceApi.stream));
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

    CUDA_DRIVER_CALL(cuInit(0));

    int num_device = 0;
    CUDA_DRIVER_CALL(cuDeviceGetCount(&num_device));
    cudaDeviceApi.num_device = num_device;

    CUDA_DRIVER_CALL(cuDeviceGetDefaultMemPool(&cudaDeviceApi.mem_pool, 0));

    cudaDeviceApi.contexts = TVM_RT_WASM_HeapMemoryAlloc(sizeof(CUstream) * num_device);
    for (int i = 0; i < num_device; ++i) {
        cuCtxCreate(cudaDeviceApi.contexts + i, 0, i);
    }
    cudaDeviceApi.stream = NULL;

    return 0;

#else
    CUDA_NOT_SUPPORTED();
#endif
}
