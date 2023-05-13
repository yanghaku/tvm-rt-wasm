/*!
 * \file device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <device/cuda_device_api.h>
#include <string.h>
#include <utils/tensor_helper.h>

#if USE_CUDA // USE_CUDA = 1

/*! \brief the cuda Device API will be a single static instance */
static CUDADeviceAPI cudaDeviceApi;

static int TVM_RT_WASM_CUDA_SetDevice(int dev_id) {
    if (unlikely(cudaDeviceApi.current_device != dev_id)) {
        cudaDeviceApi.current_device = dev_id;
        CUDA_DRIVER_CALL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
    }
    return 0;
}

static void *TVM_RT_WASM_CUDA_AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *res = NULL;
    CUDA_DRIVER_CALL_NULL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
    return res;
}

static void *TVM_RT_WASM_CUDA_AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype,
                                                  const char *mem_scope) {
    SET_ERROR_RETURN((void *)-1, "%s is not supported yet\n", __FUNCTION__);
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
    CUDA_DRIVER_CALL_NULL(cuCtxSetCurrent(cudaDeviceApi.contexts[dev_id]));
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

static int TVM_RT_WASM_CUDA_SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    return 0;
}

#ifdef CUDA_10_ONLY
#define MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE 100
typedef struct {
    void *ptr;
    size_t size;
    uint32_t is_free;
} CachedWorkspaceMemory;
static CachedWorkspaceMemory cachedWorkspaceMemory[MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE];
static uint32_t cachedWorkspaceMemorySize = 0;
#endif // CUDA_10_ONLY

static void *TVM_RT_WASM_CUDA_AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *res = NULL;

#ifdef CUDA_10_ONLY
    for (uint32_t i = 0; i < cachedWorkspaceMemorySize; ++i) {
        if (cachedWorkspaceMemory[i].size == nbytes && cachedWorkspaceMemory[i].is_free) {
            cachedWorkspaceMemory[i].is_free = 0;
            return cachedWorkspaceMemory[i].ptr;
        }
    }
    CUDA_DRIVER_CALL_NULL(cuMemAlloc((CUdeviceptr *)&res, nbytes));
    if (cachedWorkspaceMemorySize == MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE) { // cache is full
        cachedWorkspaceMemorySize = 0;
        // free the unused cached
        // todo: fix memory leak here.
        for (uint32_t i = 0; i < MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE; ++i) {
            if (!cachedWorkspaceMemory[i].is_free) {
                cachedWorkspaceMemory[cachedWorkspaceMemorySize++] = cachedWorkspaceMemory[i];
            }
        }
    }
    if (cachedWorkspaceMemorySize < MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE) {
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].is_free = 0;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].ptr = res;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize++].size = nbytes;
    }
#else
    CUDA_DRIVER_CALL_NULL(
        cuMemAllocFromPoolAsync((CUdeviceptr *)&res, nbytes, cudaDeviceApi.mem_pool, cudaDeviceApi.stream));
#endif // CUDA_10_ONLY

    return res;
}

static int TVM_RT_WASM_CUDA_FreeWorkspace(int dev_id, void *ptr) {
#ifdef CUDA_10_ONLY
    for (int i = 0; i < cachedWorkspaceMemorySize; ++i) {
        if (cachedWorkspaceMemory[i].ptr == ptr) {
            cachedWorkspaceMemory[i].is_free = 1;
            return 0;
        }
    }
    CUDA_DRIVER_CALL(cuMemFree((CUdeviceptr)ptr));
#else
    CUDA_DRIVER_CALL(cuMemFreeAsync((CUdeviceptr)ptr, cudaDeviceApi.stream));
#endif // CUDA_10_ONLY
    return 0;
}

static int TVM_RT_WASM_CUDA_Release(DeviceAPI *d) {
    if (d != (DeviceAPI *)&cudaDeviceApi)
        return -1;

#ifdef CUDA_10_ONLY
    for (int i = 0; i < cachedWorkspaceMemorySize; ++i) {
        TVM_RT_WASM_CUDA_FreeDataSpace(0, cachedWorkspaceMemory[i].ptr);
    }
#endif // CUDA_10_ONLY

    for (uint32_t i = 0; i < cudaDeviceApi.num_device; ++i) {
        cuCtxDestroy(cudaDeviceApi.contexts[i]);
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
    if (num_device <= 0) {
        SET_ERROR_RETURN(-1, "CUDA: no available devices.");
    }
    cudaDeviceApi.num_device = num_device;
    cudaDeviceApi.current_device = -1;

#ifndef CUDA_10_ONLY
    CUDA_DRIVER_CALL(cuDeviceGetDefaultMemPool(&cudaDeviceApi.mem_pool, 0));
#else
    cachedWorkspaceMemorySize = 0;
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
