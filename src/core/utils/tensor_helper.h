/*!
 * \file utils/tensor_helper.h
 * \brief the utils function for DLTensor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <device/cpu_memory.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/common.h>

/*! \brief Magic number for NDArray file */
extern const uint64_t kTVMNDArrayMagic;

/*! \brief Magic number for NDArray list file  */
extern const uint64_t kTVMNDArrayListMagic;

/*!
 * \brief parse string to DLDataType
 * @param str the source string
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
INLINE int TVM_RT_WASM_DLDataType_ParseFromString(const char *str, DLDataType *out_type) {
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
        char *tmp_str = TVM_RT_WASM_WorkplaceMemoryAlloc(strlen(str) + 1);
        strcpy(tmp_str, str);
        TVM_RT_SET_ERROR("Unsupported DLDateType: %s", tmp_str);
        TVM_RT_WASM_WorkplaceMemoryFree(tmp_str);
        return -1;
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
INLINE int64_t TVM_RT_WASM_DLTensor_GetDataSize(const int64_t *shape, int ndim) {
    int64_t size = 1;
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
INLINE uint64_t TVM_RT_WASM_DLTensor_GetDataBytes(const DLTensor *tensor) {
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
int TVM_RT_WASM_DLTensor_LoadDataFromBinary(DLTensor *tensor, const char **blob);

/*!
 * \brief parse binary and load data to tensor (only load the data)
 * @param tensor the init tensor with no data
 * @param fp the opened file struct
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadDataFromFile(DLTensor *tensor, FILE *fp);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_TENSOR_HELPER_H_INCLUDE_
