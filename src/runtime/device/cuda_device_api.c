/*!
 * \file src/runtime/device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/device/cuda_device_api.h>
#include <tvm/runtime/utils/tensor_helper.h>

#if USE_CUDA // USE_CUDA = 1

/*! \brief the cuda Device API will be a single static instance */
static CUDADeviceAPI cudaDeviceApi;

static void SetDevice(int dev_id) { CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id])); }

static void *AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
    return res;
}

static void *AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype, const char *mem_scope) {
    fprintf(stderr, "%s is not supported yet\n", __FUNCTION__);
    exit(-1);
}

static void FreeDataSpace(int dev_id, void *ptr) { CUDA_DRIVER_CALL(cuMemFree((CUdeviceptr)ptr)); }

static void CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
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

static TVMStreamHandle CreateStream(int dev_id) {
    CUstream out;
    CUDA_DRIVER_CALL(cuStreamCreate(&out, CU_STREAM_DEFAULT));
    return out;
}

static void FreeStream(int dev_id, TVMStreamHandle stream) { CUDA_DRIVER_CALL(cuStreamDestroy(stream)); }

static void StreamSync(int dev_id, TVMStreamHandle stream) { CUDA_DRIVER_CALL(cuStreamSynchronize(stream)); }

static void SetStream(int dev_id, TVMStreamHandle stream) { cudaDeviceApi.stream = stream; }

static TVMStreamHandle GetStream() { return cudaDeviceApi.stream; }

static void SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {}

static void *AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL(cuMemAllocAsync((CUdeviceptr *)&res, nbytes, cudaDeviceApi.stream));
    return res;
}

static void FreeWorkspace(int dev_id, void *ptr) {
    CUDA_DRIVER_CALL(cuMemFreeAsync((CUdeviceptr)ptr, cudaDeviceApi.stream));
}

static int Release(DeviceAPI *d) {
    if (d != (DeviceAPI *)&cudaDeviceApi)
        return -1;
    for (uint32_t i = 0; i < cudaDeviceApi.num_device; ++i) {
        CUDA_DRIVER_CALL(cuCtxDestroy(cudaDeviceApi.contexts[i]));
    }
    DLDevice cpu = {kDLCPU, 0};
    return TVMDeviceFreeDataSpace(cpu, cudaDeviceApi.contexts);
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

    cudaDeviceApi.SetDevice = SetDevice;
    cudaDeviceApi.AllocDataSpace = AllocDataSpace;
    cudaDeviceApi.AllocDataSpaceScope = AllocDataSpaceScope;
    cudaDeviceApi.FreeDataSpace = FreeDataSpace;
    cudaDeviceApi.CopyDataFromTo = CopyDataFromTo;
    cudaDeviceApi.CreateStream = CreateStream;
    cudaDeviceApi.FreeStream = FreeStream;
    cudaDeviceApi.StreamSync = StreamSync;
    cudaDeviceApi.SetStream = SetStream;
    cudaDeviceApi.GetStream = GetStream;
    cudaDeviceApi.SyncStreamFromTo = SyncStreamFromTo;
    cudaDeviceApi.AllocWorkspace = AllocWorkspace;
    cudaDeviceApi.FreeWorkspace = FreeWorkspace;
    cudaDeviceApi.Release = Release;

    int num_device = 0;
    CUDA_DRIVER_CALL(cuDeviceGetCount(&num_device));
    cudaDeviceApi.num_device = num_device;

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(CUstream) * num_device, 0, no_type, (void **)&cudaDeviceApi.contexts);
    for (int i = 0; i < num_device; ++i) {
        cuCtxCreate(cudaDeviceApi.contexts + i, 0, i);
    }
    cudaDeviceApi.stream = NULL;

    return 0;

#else
    CUDA_NOT_SUPPORTED();
#endif
}
