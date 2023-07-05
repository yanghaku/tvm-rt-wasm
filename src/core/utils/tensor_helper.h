/**
 * @file utils/tensor_helper.h
 * @brief the utils function for DLTensor
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <utils/common.h>
#include <utils/stream_reader.h>

/** @brief Magic number for NDArray file */
#define kTVMNDArrayMagic (0xDD5E40F096B4A13FUL)

/** @brief Magic number for NDArray list file  */
#define kTVMNDArrayListMagic (0xF7E58D4F05049CB7UL)

/**
 * @brief get data number of tensor.
 * @param shape the shape of tensor.
 * @param ndim the number of dim.
 * @return result
 */
INLINE int64_t TVM_RT_WASM_DLTensor_GetDataSize(const int64_t *shape, int ndim) {
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

/**
 * @brief Get data bytes number of tensor.
 * @param tensor the tensor pointer.
 * @return result
 */
INLINE uint64_t TVM_RT_WASM_DLTensor_GetDataBytes(const DLTensor *tensor) {
    size_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        size *= tensor->shape[i];
    }
    size *= (tensor->dtype.bits * tensor->dtype.lanes + 7) / 8;
    return size;
}

/**
 * @brief Parse binary and load data to tensor.
 * @param tensor the tensor with no data or with data.
 * @param reader The stream reader instance.
 * @note If shape is NULL, it will alloc memory for shape.
 * @note If data is NULL, it will alloc memory for data.
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadFromReader(DLTensor *tensor, StreamReader *reader);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_
