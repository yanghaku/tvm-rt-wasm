/*!
 * \file runtime/utils/tensor.h
 * \brief the utils function for DLTensor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_TENSOR_H
#define TVM_RT_TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

/*! \brief Magic number for NDArray file */
static const uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

/*! \brief Magic number for NDArray list file  */
static const uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief copy the data memory for DLTensor
 * @param data_from src DLTensor
 * @param data_to dst DLTensor
 * @param stream stream handle
 * @return 0 if successful
 */
static inline int DLTensor_CopyFromTo(const DLTensor *data_from, DLTensor *data_to, TVMStreamHandle *stream) {
    return 0;
}

/*!
 * \brief parse string to DLDataType
 * @param str the source string
 * @param str_len source string length
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
static inline int DLDataType_ParseFromString(const char *str, uint32_t str_len, DLDataType *out_type) { return 0; }

/*!
 * \brief get data number of Tensor
 * @param shape the shape of tensor
 * @param ndim the number of dim
 * @return result
 */
static inline uint64_t DLTensor_GetDataSize(const uint64_t *shape, int ndim) {
    uint64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    return size;
}

/*!
 * \brief get data bytes number of tensor
 * @param tensor the tensor pointer
 * @return result
 */
static inline uint64_t DLTensor_GetDataBytes(const DLTensor *tensor) {
    size_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        size *= tensor->shape[i];
    }
    size *= (tensor->dtype.bits * tensor->dtype.lanes + 7) / 8;
    return size;
}

/*!
 * \brief parse binary and load data to tensor
 * @param tensor the init tensor with no data
 * @param blob the binary
 * @return 0 if successful
 */
static inline int DLTensor_LoadDataFromBinary(DLTensor *tensor, const char **blob) { return 0; }

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_TENSOR_H
