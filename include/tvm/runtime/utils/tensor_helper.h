/*!
 * \file runtime/utils/tensor_helper.h
 * \brief the utils function for DLTensor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_TENSOR_HELPER_H
#define TVM_RT_TENSOR_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/utils/common.h>
#include <tvm/runtime/utils/json.h>

/*! \brief Magic number for NDArray file */
const uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

/*! \brief Magic number for NDArray list file  */
const uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief parse string to DLDataType
 * @param str the source string
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
INLINE int DLDataType_ParseFromString(const char *str, DLDataType *out_type) {
    if (*str == 0) { // void
        out_type->code = kDLOpaqueHandle;
        out_type->lanes = 0;
        out_type->bits = 0;
        return 0;
    }
    out_type->lanes = 1;
    if (!memcmp(str, "int", 3)) {
        out_type->code = kDLInt;
        out_type->bits = 32;
        str += 3;
    } else if (!memcmp(str, "uint", 4) || !memcmp(str, "bool", 4)) {
        out_type->code = kDLUInt;
        out_type->bits = 32;
        str += 4;
    } else if (!memcmp(str, "float", 5)) {
        out_type->code = kDLFloat;
        out_type->bits = 32;
        str += 4;
    } else if (!memcmp(str, "handle", 6)) {
        out_type->code = kDLOpaqueHandle;
        out_type->bits = 64;
        str += 6;
    } else {
        SET_ERROR_RETURN(-1, "unsupported DLDateType: %s", str);
    }
    if (isdigit1to9(*str)) {
        int num = 0;
        while (isdigit0to9(*str)) {
            (num) = ((num) << 3) + ((num) << 1) + (*str++) - '0';
        }
        out_type->bits = num;
    }
    return 0;
}

/*!
 * \brief get data number of Tensor
 * @param shape the shape of tensor
 * @param ndim the number of dim
 * @return result
 */
INLINE uint64_t DLTensor_GetDataSize(const uint64_t *shape, int ndim) {
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
INLINE uint64_t DLTensor_GetDataBytes(const DLTensor *tensor) {
    size_t size = 1;
    for (int i = 0; i < tensor->ndim; ++i) {
        size *= tensor->shape[i];
    }
    size *= (tensor->dtype.bits * tensor->dtype.lanes + 7) / 8;
    return size;
}

/*!
 * \brief parse binary and load data to tensor (only load the data)
 * @param tensor the init tensor with no data
 * @param blob the binary
 * @return 0 if successful
 */
INLINE int DLTensor_LoadDataFromBinary(DLTensor *tensor, const char **blob) {
    uint64_t header;
    memcpy(&header, *blob, sizeof(header));
    *blob += sizeof(header);
    if (unlikely(header != kTVMNDArrayMagic)) {
        SET_ERROR_RETURN(-1, "Invalid DLTensor file Magic number: %llu\n", header);
    }
    *blob += sizeof(uint64_t); // reserved
    *blob += sizeof(DLDevice); // DLDevice

    if (unlikely(memcmp(&tensor->ndim, *blob, sizeof(int)))) { // ndim
        SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, given %d", tensor->ndim, *(int *)blob);
    }
    *blob += sizeof(int); // ndim

    //    if (unlikely(memcmp(&tensor->dtype, *blob, sizeof(DLDataType)))) { // DLDateType
    //    }
    *blob += sizeof(DLDataType); // DLDataType

    for (int i = 0; i < tensor->ndim; ++i) { // shapes
        if (unlikely(tensor->shape[i] != *(int64_t *)(*blob))) {
            SET_ERROR_RETURN(-1, "Invalid DLTensor shape: expect shape[%d] = %lld, but given %lld\n", i,
                             tensor->shape[i], *(int64_t *)(*blob));
        }
        *blob += sizeof(int64_t); // shape
    }

    int64_t byte_size;
    memcpy(&byte_size, *blob, sizeof(byte_size));
    int64_t tensor_size = (int64_t)DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        SET_ERROR_RETURN(-1, "Invalid DLTensor ata byte size: expect %llu, but given %llu\n", tensor_size, byte_size);
    }
    *blob += sizeof(byte_size); // byte_size

    DLDevice cpu = {kDLCPU, 0};
    DLTensor src_tensor = {
        .ndim = tensor->ndim,
        .shape = tensor->shape,
        .dtype = tensor->dtype,
        .device = cpu,
        .data = (void *)*blob,
    };

    // copy data
    int status = TVMDeviceCopyDataFromTo(&src_tensor, tensor, NULL);
    *blob += byte_size;

    return status;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_TENSOR_HELPER_H
