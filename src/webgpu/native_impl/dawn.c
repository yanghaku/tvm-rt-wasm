/*!
 * \file webgpu/native_impl/dawn.c
 * \brief link with dawn library implementation.
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#if USE_WEBGPU && !defined(__EMSCRIPTEN__) // USE_WEBGPU = 1 && !defined(__EMSCRIPTEN__)

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <utils/common.h>
#include <webgpu/webgpu_c_api.h>

#define STRINGIFY(a) #a
#define WGPU_Native_To_Dawn(obj)                                                                                       \
    _Pragma(STRINGIFY(weak wgpu##obj##Drop));                                                                          \
    typedef struct WGPU##obj##Impl *WGPU##obj;                                                                         \
    extern void wgpu##obj##Release(WGPU##obj);                                                                         \
    void wgpu##obj##Drop(WGPU##obj arg) { wgpu##obj##Release(arg); }

WGPU_Native_To_Dawn(Instance);
WGPU_Native_To_Dawn(Adapter);
WGPU_Native_To_Dawn(BindGroup);
WGPU_Native_To_Dawn(ShaderModule);
WGPU_Native_To_Dawn(ComputePipeline);

typedef struct WGPUQueueImpl *WGPUQueue;
typedef struct WGPUDeviceImpl *WGPUDevice;
typedef struct WGPUWrappedSubmissionIndex WGPUWrappedSubmissionIndex;
typedef enum WGPUQueueWorkDoneStatus {
    WGPUQueueWorkDoneStatus_Success = 0x00000000,
    WGPUQueueWorkDoneStatus_Error = 0x00000001,
    WGPUQueueWorkDoneStatus_Unknown = 0x00000002,
    WGPUQueueWorkDoneStatus_DeviceLost = 0x00000003,
    WGPUQueueWorkDoneStatus_Force32 = 0x7FFFFFFF
} WGPUQueueWorkDoneStatus;
typedef void (*WGPUQueueWorkDoneCallback)(WGPUQueueWorkDoneStatus status, void *userdata);

// the functions in dawn
extern void wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, uint64_t signalValue, WGPUQueueWorkDoneCallback callback,
                                         void *userdata);
extern WGPUQueue wgpuDeviceGetQueue(WGPUDevice device);

typedef struct {
    pthread_mutex_t ready_mutex;
    pthread_cond_t ready_cond;
    int is_ready;
} DeviceSyncData;

#ifndef WEBGPU_MAX_DEVICES
#define WEBGPU_MAX_DEVICES 10
#endif // WEBGPU_MAX_DEVICES

static DeviceSyncData deviceSyncData[WEBGPU_MAX_DEVICES];
static WGPUDevice devices[WEBGPU_MAX_DEVICES];
static uint32_t currentNumDevices;

static __attribute__((destructor)) void dawn_device_callback_destructor() {
    for (uint32_t i = 0; i < currentNumDevices; ++i) {
        pthread_cond_destroy(&deviceSyncData[i].ready_cond);
        pthread_mutex_destroy(&deviceSyncData[i].ready_mutex);
    }
    currentNumDevices = 0;
}

static void queue_submit_done_callback(WGPUQueueWorkDoneStatus status, void *userdata) {
    DeviceSyncData *data = (DeviceSyncData *)userdata;
    pthread_mutex_lock(&data->ready_mutex);
    data->is_ready = 1;
    pthread_cond_signal(&data->ready_cond);
    pthread_mutex_unlock(&data->ready_mutex);
}

#pragma weak wgpuDevicePoll
bool wgpuDevicePoll(WGPUDevice device, bool wait, WGPUWrappedSubmissionIndex const *wrappedSubmissionIndex) {
    uint32_t device_id = 0;
    while (device_id < currentNumDevices) {
        if (devices[device_id] == device) {
            break;
        }
    }
    DeviceSyncData *data = deviceSyncData + device_id;

    if (unlikely(device_id == currentNumDevices)) { // add new device
        if (unlikely(currentNumDevices == (WEBGPU_MAX_DEVICES))) {
            // todo
        }
        ++currentNumDevices;

        devices[device_id] = device;
        data->is_ready = 1;
        pthread_mutex_init(&data->ready_mutex, NULL);
        pthread_cond_init(&data->ready_cond, NULL);

        WGPUQueue queue = wgpuDeviceGetQueue(device);
        wgpuQueueOnSubmittedWorkDone(queue, 0, queue_submit_done_callback, (void *)data);
    } else { // do sync
        // block current thread to wait
        pthread_mutex_lock(&data->ready_mutex);
        while (!data->is_ready) {
            pthread_cond_wait(&data->ready_cond, &data->ready_mutex);
        }
        pthread_mutex_unlock(&data->ready_mutex);
    }
    return 0;
}

void wgpuAddTasksToQueue(WGPUDevice device) {
    for (uint32_t device_id = 0; device_id < currentNumDevices; ++device_id) {
        if (devices[device_id] == device) {
            DeviceSyncData *data = deviceSyncData + device_id;
            pthread_mutex_lock(&data->ready_mutex);
            data->is_ready = 0;
            pthread_mutex_unlock(&data->ready_mutex);
        }
    }
}

#endif // // USE_WEBGPU = 1 && !defined(__EMSCRIPTEN__)
