/*!
 * \file src/runtime/device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <stdlib.h>
#include <tvm/runtime/device/cuda_device_api.h>

#if USE_CUDA // USE_CUDA = 1

/*! \brief the cuda Device API will be a single static instance */
static CUDADeviceAPI cudaDeviceApi;

static void SetDevice(int dev_id) {}
static void *AllocDataSpace(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint) {}
static void *AllocDataSpaceScope(int dev_id, int ndim, const int64_t *shape, DLDataType dtype, const char *mem_scope) {}
static void FreeDataSpace(int dev_id, void *ptr) {}
static void CopyDataFromTo(DLTensor *from, DLTensor *to, TVMStreamHandle stream) {}
TVMStreamHandle CreateStream(int dev_id) {}
static void FreeStream(int dev_id, TVMStreamHandle stream) {}
static void StreamSync(int dev_id, TVMStreamHandle stream) {}
static void SetStream(int dev_id, TVMStreamHandle stream) {}
static void SyncStreamFromTo(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst) {}
static void *AllocWorkspace(int dev_id, size_t nbytes, DLDataType type_hint) {}
static void FreeWorkspace(int dev_id, void *ptr) {}
static int Release(DeviceAPI *d) { return 0; }

#endif

/*!
 * \brief create a instance of cuda device api
 * @param out the pointer to receive instance
 * @return 0 if successful
 */
int CUDADeviceAPICreate(CUDADeviceAPI **out) {

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
    cudaDeviceApi.SyncStreamFromTo = SyncStreamFromTo;
    cudaDeviceApi.AllocWorkspace = AllocWorkspace;
    cudaDeviceApi.FreeWorkspace = FreeWorkspace;
    cudaDeviceApi.Release = Release;

    CUDA_CALL(cudaGetDeviceCount((int *)&(*cudaDeviceApi)->num_device));
    // todo: init CUcontext
    return 0;

#else
    fprintf(stderr, "CUDA library is not supported! you can compile from source and set USE_CUDA option ON\n");
    exit(-1);
#endif
}
