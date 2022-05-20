/*!
 * \file wasmer-cuda-share.h
 * \author Bo Yang (bo.yang\@smail.nju.edu.cn)
 * \brief Interface for cuda memory share in wasmer
 */

#ifndef WASMER_CUDA_SHARE_WASMER_CUDA_SHARE_H
#define WASMER_CUDA_SHARE_WASMER_CUDA_SHARE_H

#include <cuda.h>

/*------------------------------------ Module Management -------------------------------------------------------------*/

CUresult cuModuleLoadShared(CUmodule *module, const char *filename, const char *shared_key);

CUresult cuModuleLoadDataShared(CUmodule *module, const void *image, const char *shared_key);

CUresult cuModuleLoadDataExShared(CUmodule *module, const void *image, unsigned int num_options, CUjit_option *options,
                                  void **option_values, const char *shared_key);

CUresult cuModuleLoadFatBinaryShared(CUmodule *module, const void *fat_cu_bin);

/*------------------------------------ Memory Management -------------------------------------------------------------*/

CUresult cuMemAllocShared(CUdeviceptr *d_ptr, size_t bytesize, const char *shared_key);

CUresult cuMemAllocAsyncShared(CUdeviceptr *d_ptr, size_t bytesize, CUstream hStream, const char *shared_key);

CUresult cuMemAllocFromPoolAsyncShared(CUdeviceptr *d_ptr, size_t bytesize, CUmemoryPool pool, CUstream hStream,
                                       const char *shared_key);

#endif // WASMER_CUDA_SHARE_WASMER_CUDA_SHARE_H
