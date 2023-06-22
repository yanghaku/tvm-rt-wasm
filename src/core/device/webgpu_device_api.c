/*!
 * \file device/webgpu_device_api.c
 * \brief implement for webgpu device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <device/webgpu_device_api.h>
#include <string.h>
#include <utils/tensor_helper.h>

#if USE_WEBGPU // USE_WEBGPU = 1

/*! \brief the webgpu Device API will be a single static instance */
static WebGPUDeviceAPI webGPUDeviceAPI;

static int TVM_RT_WASM_WebGPU_SetDevice(int dev_id) { return 0; }

static void *TVM_RT_WASM_WebGPU_AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *res = NULL;
    WGPU_CALL_NULL(WGPU_MemoryAlloc(webGPUDeviceAPI.device, (WGPU_Memory *)&res, nbytes));
    return res;
}

static void *TVM_RT_WASM_WebGPU_AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype,
                                                    const char *mem_scope) {
    SET_ERROR_RETURN((void *)-1, "%s is not supported yet\n", __FUNCTION__);
}

static int TVM_RT_WASM_WebGPU_FreeDataSpace(int dev_id, void *ptr) {
    WGPU_CALL(WGPU_MemoryFree((WGPU_Memory)ptr));
    return 0;
}

static int TVM_RT_WASM_WebGPU_CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
    uint64_t bytes = TVM_RT_WASM_DLTensor_GetDataBytes(from);
    uint64_t byte_check = TVM_RT_WASM_DLTensor_GetDataBytes(to);
    if (unlikely(bytes != byte_check)) {
        SET_ERROR_RETURN(-1, "Error: data copy size is diff, from=%lld and to=%lld\n", bytes, byte_check);
    }
    if (from->device.device_type == kDLCPU) {
        if (to->device.device_type == kDLCPU) {
            memcpy(to->data, from->data, bytes);
        } else if (to->device.device_type == kDLWebGPU) {
            WGPU_CALL(
                WGPU_MemoryCopyHtoD((WGPU_Memory)to->data, to->byte_offset, from->data, from->byte_offset, bytes));
        } else {
            SET_ERROR_RETURN(-1, "Error: unsupported data copy!\n");
        }
    } else if (from->device.device_type == kDLWebGPU) {
        if (to->device.device_type == kDLCPU) {
            WGPU_CALL(
                WGPU_MemoryCopyDtoH(to->data, to->byte_offset, (WGPU_Memory)from->data, from->byte_offset, bytes));
        } else if (to->device.device_type == kDLWebGPU) {
            WGPU_CALL(WGPU_MemoryCopyDtoD((WGPU_Memory)to->data, to->byte_offset, (WGPU_Memory)from->data,
                                          from->byte_offset, bytes));
        } else {
            SET_ERROR_RETURN(-1, "Error: unsupported data copy!\n");
        }
    }
    return 0;
}

static TVMStreamHandle TVM_RT_WASM_WebGPU_CreateStream(int dev_id) {
    TVMStreamHandle out = NULL;
    // todo
    return out;
}

static int TVM_RT_WASM_WebGPU_FreeStream(int dev_id, TVMStreamHandle stream) {
    // todo
    return 0;
}

static int TVM_RT_WASM_WebGPU_StreamSync(int dev_id, TVMStreamHandle stream) {
    // todo
    return 0;
}

static int TVM_RT_WASM_WebGPU_SetStream(int dev_id, TVMStreamHandle stream) {
    // todo
    return 0;
}

static TVMStreamHandle TVM_RT_WASM_WebGPU_GetStream() {
    // the webgpu has no stream, now it return the current device
    return (TVMStreamHandle)webGPUDeviceAPI.device;
}

static int TVM_RT_WASM_WebGPU_SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    SET_ERROR_RETURN(-1, "%s is not supported yet\n", __FUNCTION__);
}

#define MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE 100
typedef struct {
    void *ptr;
    size_t size;
    uint32_t is_free;
} CachedWorkspaceMemory;
static CachedWorkspaceMemory cachedWorkspaceMemory[MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE];
static uint32_t cachedWorkspaceMemorySize = 0;

static void *TVM_RT_WASM_WebGPU_AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *res = NULL;

    for (uint32_t i = 0; i < cachedWorkspaceMemorySize; ++i) {
        if (cachedWorkspaceMemory[i].size == nbytes && cachedWorkspaceMemory[i].is_free) {
            cachedWorkspaceMemory[i].is_free = 0;
            return cachedWorkspaceMemory[i].ptr;
        }
    }
    WGPU_CALL_NULL(WGPU_MemoryAlloc(webGPUDeviceAPI.device, (WGPU_Memory *)&res, nbytes));

    if (cachedWorkspaceMemorySize < MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE) {
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].is_free = 0;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].ptr = res;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize++].size = nbytes;
    }

    return res;
}

static int TVM_RT_WASM_WebGPU_FreeWorkspace(int dev_id, void *ptr) {
    for (int i = cachedWorkspaceMemorySize - 1; i >= 0; --i) {
        if (cachedWorkspaceMemory[i].ptr == ptr) {
            cachedWorkspaceMemory[i].is_free = 1;
            return 0;
        }
    }
    WGPU_CALL(WGPU_MemoryFree(ptr));
    return 0;
}

static int TVM_RT_WASM_WebGPU_Release(DeviceAPI *d) {
    if (d != (DeviceAPI *)&webGPUDeviceAPI)
        return -1;

    for (int i = 0; i < cachedWorkspaceMemorySize; ++i) {
        TVM_RT_WASM_WebGPU_FreeDataSpace(0, cachedWorkspaceMemory[i].ptr);
    }

    WGPU_CALL(WGPU_DeviceFree(webGPUDeviceAPI.device));
    return 0;
}

#endif // USE_WEBGPU

/*!
 * \brief create a instance of webgpu device api
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int TVM_RT_WASM_WebGPUDeviceAPICreate(WebGPUDeviceAPI **out) {

#if USE_WEBGPU // USE_WEBGPU = 1

    *out = &webGPUDeviceAPI;

    webGPUDeviceAPI.SetDevice = TVM_RT_WASM_WebGPU_SetDevice;
    webGPUDeviceAPI.AllocDataSpace = TVM_RT_WASM_WebGPU_AllocDataSpace;
    webGPUDeviceAPI.AllocDataSpaceScope = TVM_RT_WASM_WebGPU_AllocDataSpaceScope;
    webGPUDeviceAPI.FreeDataSpace = TVM_RT_WASM_WebGPU_FreeDataSpace;
    webGPUDeviceAPI.CopyDataFromTo = TVM_RT_WASM_WebGPU_CopyDataFromTo;
    webGPUDeviceAPI.CreateStream = TVM_RT_WASM_WebGPU_CreateStream;
    webGPUDeviceAPI.FreeStream = TVM_RT_WASM_WebGPU_FreeStream;
    webGPUDeviceAPI.StreamSync = TVM_RT_WASM_WebGPU_StreamSync;
    webGPUDeviceAPI.SetStream = TVM_RT_WASM_WebGPU_SetStream;
    webGPUDeviceAPI.GetStream = TVM_RT_WASM_WebGPU_GetStream;
    webGPUDeviceAPI.SyncStreamFromTo = TVM_RT_WASM_WebGPU_SyncStreamFromTo;
    webGPUDeviceAPI.AllocWorkspace = TVM_RT_WASM_WebGPU_AllocWorkspace;
    webGPUDeviceAPI.FreeWorkspace = TVM_RT_WASM_WebGPU_FreeWorkspace;
    webGPUDeviceAPI.Release = TVM_RT_WASM_WebGPU_Release;

    cachedWorkspaceMemorySize = 0;

    SET_TIME(t0)

    WGPU_CALL(WGPU_DeviceGet(&webGPUDeviceAPI.device));

    SET_TIME(t1)

    DURING_PRINT(t1, t0, "init webgpu device time");
    return 0;

#else
    WebGPU_NOT_SUPPORTED();
#endif // USE_WEBGPU
}
