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
#include <utils/binary_reader.h>
#include <utils/common.h>
#include <utils/stream_reader.h>

/** @brief Magic number for NDArray file */
#define kTVMNDArrayMagic (UINT64_C(0xDD5E40F096B4A13F))

/** @brief Magic number for NDArray list file  */
#define kTVMNDArrayListMagic (UINT64_C(0xF7E58D4F05049CB7))

/**
 * @brief Get data bytes number of tensor.
 * @param shape DLTensor shape
 * @param ndim DLTensor ndim
 * @param data_type DLTensor data type
 * @return result
 */
INLINE size_t TVM_RT_WASM_DLTensor_GetDataBytes(const int64_t *shape, int ndim,
                                                DLDataType data_type) {
    size_t size = ((size_t)data_type.bits * data_type.lanes + 7) / 8;
    for (int32_t i = 0; i < ndim; ++i) {
        size *= (size_t)shape[i];
    }
    return size;
}

/**
 * @brief Parse and load stream data to tensor, this function will copy data to tensor.
 * @param tensor the tensor with no data or with data.
 * @param reader The stream reader instance.
 * @note If shape is NULL, it will alloc memory for shape.
 * @note If data is NULL, it will alloc memory for data.
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadFromStream(DLTensor *tensor, StreamReader *reader);

/**
 * @brief Parse and load binary blob to tensor, this function **will not** copy shape and data.
 * @param tensor the tensor instance (no shape and no data).
 * @param reader The binary reader instance.
 * @note The function will not copy shape and data!
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadFromBinary(DLTensor *tensor, BinaryReader *reader);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_
