/*!
 * \file src/runtime/device/cpu device_api.c
 * \brief implement for cpu device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device/cpu_device_api.h>
#include <tvm/runtime/utils/common.h>
#include <tvm/runtime/utils/tensor_helper.h>

/*! \brief the cpu device API will be a static instance variable */
static CPUDeviceAPI cpuDeviceApi;

static void SetDevice(int dev_id) {}

static void *AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {
    void *p = malloc(nbytes);
    if (unlikely(p == NULL)) {
        fprintf(stderr, "allocate memory Fail! (allocate size = %zu\n", nbytes);
        exit(-1);
    }
    return p;
}

static void *AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype, const char *mem_scope) {
    fprintf(stderr, "%s is not supported yet\n", __FUNCTION__);
    exit(-1);
}

static void FreeDataSpace(int dev_id, void *ptr) { free(ptr); }

static void CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {
    uint64_t size_from = DLTensor_GetDataBytes(from);
    uint64_t size_to = DLTensor_GetDataBytes(to);
    if (unlikely(size_from != size_to)) {
        fprintf(stderr, "copy memory data byte size is not same: from(%llu) != to(%llu)", size_from, size_to);
        exit(-1);
    }
    memcpy(to->data, from->data, size_to);
}

static TVMStreamHandle CreateStream(int dev_id) { return NULL; }

static void FreeStream(int dev_id, TVMStreamHandle stream) {}

static void StreamSync(int dev_id, TVMStreamHandle stream) {}

static void SetStream(int dev_id, TVMStreamHandle stream) {}

static TVMStreamHandle GetStream() { return NULL; }

static void SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {}

static void *AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {
    void *p = malloc(nbytes);
    if (unlikely(p == NULL)) {
        fprintf(stderr, "allocate memory Fail! (allocate size = %zu\n", nbytes);
        exit(-1);
    }
    return p;
}

static void FreeWorkspace(int dev_id, void *ptr) { free(ptr); }

static int Release(DeviceAPI *d) { return 0; }

/*!
 * \brief create the cpu_device_api instance
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int CPUDeviceAPICreate(CPUDeviceAPI **out) {
    *out = &cpuDeviceApi;

    cpuDeviceApi.SetDevice = SetDevice;
    cpuDeviceApi.AllocDataSpace = AllocDataSpace;
    cpuDeviceApi.AllocDataSpaceScope = AllocDataSpaceScope;
    cpuDeviceApi.FreeDataSpace = FreeDataSpace;
    cpuDeviceApi.CopyDataFromTo = CopyDataFromTo;
    cpuDeviceApi.CreateStream = CreateStream;
    cpuDeviceApi.FreeStream = FreeStream;
    cpuDeviceApi.StreamSync = StreamSync;
    cpuDeviceApi.SetStream = SetStream;
    cpuDeviceApi.GetStream = GetStream;
    cpuDeviceApi.SyncStreamFromTo = SyncStreamFromTo;
    cpuDeviceApi.AllocWorkspace = AllocWorkspace;
    cpuDeviceApi.FreeWorkspace = FreeWorkspace;
    cpuDeviceApi.Release = Release;

    return 0;
}
