/**
 * @file device/webgpu_device_api.c
 * @brief implement for webgpu device api
 */

#include <device/device_api.h>
#include <webgpu_common.h>

/** @brief WebGPUDeviceAPI implement the interface DeviceAPI */
typedef struct WebGPUDeviceAPI {
    DEVICE_API_INTERFACE

    WGPU_Device device;

} WebGPUDeviceAPI;

/** @brief the webgpu Device API will be a single static instance */
static WebGPUDeviceAPI webGPUDeviceAPI;

static int TVM_RT_WASM_WebGPU_SetDevice(int dev_id) {
    (void)dev_id;
    return 0;
}

static void *TVM_RT_WASM_WebGPU_AllocDataSpace(int dev_id, size_t nbytes) {
    (void)dev_id;

    void *res = NULL;
    int status = WGPU_MemoryAlloc(webGPUDeviceAPI.device, (WGPU_Memory *)&res, nbytes);
    if (unlikely(status)) {
        return NULL;
    }
    return res;
}

static int TVM_RT_WASM_WebGPU_FreeDataSpace(int dev_id, void *ptr) {
    (void)dev_id;
    WGPU_CALL(WGPU_MemoryFree((WGPU_Memory)ptr));
    return 0;
}

static int TVM_RT_WASM_WebGPU_CopyDataFromDeviceToCPU(const void *from, void *to, size_t nbytes,
                                                      size_t from_offset, size_t to_offset,
                                                      TVMStreamHandle stream, int from_dev_id) {
    (void)stream;
    (void)from_dev_id;
    WGPU_CALL(WGPU_MemoryCopyDtoH(to, to_offset, (WGPU_Memory)from, from_offset, nbytes));
    return 0;
}

static int TVM_RT_WASM_WebGPU_CopyDataFromCPUToDevice(const void *from, void *to, size_t nbytes,
                                                      size_t from_offset, size_t to_offset,
                                                      TVMStreamHandle stream, int to_dev_id) {
    (void)stream;
    (void)to_dev_id;
    WGPU_CALL(WGPU_MemoryCopyHtoD((WGPU_Memory)to, to_offset, from, from_offset, nbytes));
    return 0;
}

static int TVM_RT_WASM_WebGPU_CopyDataFromDeviceToDevice(const void *from, void *to, size_t nbytes,
                                                         size_t from_offset, size_t to_offset,
                                                         TVMStreamHandle stream, int from_dev_id,
                                                         int to_dev_id) {
    (void)stream;
    (void)from_dev_id;
    (void)to_dev_id;
    WGPU_CALL(
        WGPU_MemoryCopyDtoD((WGPU_Memory)to, to_offset, (WGPU_Memory)from, from_offset, nbytes));
    return 0;
}

static TVMStreamHandle TVM_RT_WASM_WebGPU_CreateStream(int dev_id) {
    (void)dev_id;
    TVM_RT_NOT_IMPLEMENT(NULL);
}

static int TVM_RT_WASM_WebGPU_FreeStream(int dev_id, TVMStreamHandle stream) {
    (void)dev_id;
    (void)stream;
    TVM_RT_NOT_IMPLEMENT(-1);
}

static int TVM_RT_WASM_WebGPU_StreamSync(int dev_id, TVMStreamHandle stream) {
    (void)dev_id;
    (void)stream;
    TVM_RT_NOT_IMPLEMENT(-1);
}

static int TVM_RT_WASM_WebGPU_SetStream(int dev_id, TVMStreamHandle stream) {
    (void)dev_id;
    (void)stream;
    TVM_RT_NOT_IMPLEMENT(-1);
}

static TVMStreamHandle TVM_RT_WASM_WebGPU_GetStream() {
    // the webgpu has no stream, now it returns the current device
    return (TVMStreamHandle)webGPUDeviceAPI.device;
}

#define MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE 100
typedef struct {
    void *ptr;
    size_t size;
    uint32_t is_free;
} CachedWorkspaceMemory;
static CachedWorkspaceMemory cachedWorkspaceMemory[MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE];
static int cachedWorkspaceMemorySize = 0;

static void *TVM_RT_WASM_WebGPU_AllocWorkspace(int dev_id, size_t nbytes) {
    (void)dev_id;
    void *res = NULL;

    for (int i = 0; i < cachedWorkspaceMemorySize; ++i) {
        if (cachedWorkspaceMemory[i].size == nbytes && cachedWorkspaceMemory[i].is_free) {
            cachedWorkspaceMemory[i].is_free = 0;
            return cachedWorkspaceMemory[i].ptr;
        }
    }
    int status = WGPU_MemoryAlloc(webGPUDeviceAPI.device, (WGPU_Memory *)&res, nbytes);
    if (unlikely(status)) {
        return NULL;
    }

    if (cachedWorkspaceMemorySize < MAX_CACHED_WORKSPACE_MEMORY_ELEMENT_SIZE) {
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].is_free = 0;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize].ptr = res;
        cachedWorkspaceMemory[cachedWorkspaceMemorySize++].size = nbytes;
    }

    return res;
}

static int TVM_RT_WASM_WebGPU_FreeWorkspace(int dev_id, void *ptr) {
    (void)dev_id;
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

/**
 * @brief Create a instance of WebGPU device api.
 * @param out The pointer to save instance.
 * @return 0 if successful
 */
int TVM_RT_WASM_WebGPUDeviceAPICreate(DeviceAPI **out) {
    *out = (DeviceAPI *)&webGPUDeviceAPI;

    webGPUDeviceAPI.SetDevice = TVM_RT_WASM_WebGPU_SetDevice;
    webGPUDeviceAPI.AllocDataSpace = TVM_RT_WASM_WebGPU_AllocDataSpace;
    webGPUDeviceAPI.FreeDataSpace = TVM_RT_WASM_WebGPU_FreeDataSpace;
    webGPUDeviceAPI.CopyDataFromCPUToDevice = TVM_RT_WASM_WebGPU_CopyDataFromCPUToDevice;
    webGPUDeviceAPI.CopyDataFromDeviceToCPU = TVM_RT_WASM_WebGPU_CopyDataFromDeviceToCPU;
    webGPUDeviceAPI.CopyDataFromDeviceToDevice = TVM_RT_WASM_WebGPU_CopyDataFromDeviceToDevice;
    webGPUDeviceAPI.CreateStream = TVM_RT_WASM_WebGPU_CreateStream;
    webGPUDeviceAPI.FreeStream = TVM_RT_WASM_WebGPU_FreeStream;
    webGPUDeviceAPI.StreamSync = TVM_RT_WASM_WebGPU_StreamSync;
    webGPUDeviceAPI.SetStream = TVM_RT_WASM_WebGPU_SetStream;
    webGPUDeviceAPI.GetStream = TVM_RT_WASM_WebGPU_GetStream;
    webGPUDeviceAPI.AllocWorkspace = TVM_RT_WASM_WebGPU_AllocWorkspace;
    webGPUDeviceAPI.FreeWorkspace = TVM_RT_WASM_WebGPU_FreeWorkspace;
    webGPUDeviceAPI.Release = TVM_RT_WASM_WebGPU_Release;

    cachedWorkspaceMemorySize = 0;

    WGPU_CALL(WGPU_DeviceGet(&webGPUDeviceAPI.device));
    return 0;
}
