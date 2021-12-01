/*!
 * \file src/runtime/device/cuda_device_api.c
 * \brief implement for cuda device api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/device/cuda_device_api.h>

/*!
 * \brief create a instance of cuda device api
 * @param cudaDeviceApi the pointer to receive instance
 * @return 0 if successful
 */
int CUDADeviceAPICreate(CUDADeviceAPI **cudaDeviceApi) {

#if USE_CUDA // USE_CUDA = 1

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(cudaDeviceApi), 0, no_type, (void **)&cudaDeviceApi);

    CUDA_CALL(cudaGetDeviceCount(&(*cudaDeviceApi)->num_device));
    // todo: init CUcontext
    return 0;

#else
    fprintf(stderr, "CUDA library is not supported! you can compile from source and set USE_CUDA option ON\n");
    exit(-1);
#endif
}
