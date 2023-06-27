/*!
 * \file utils/tensor_helper.c
 * \brief the utils function implementation for DLTensor.
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#include <utils/tensor_helper.h>

/*! \brief Magic number for NDArray file */
const uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

/*! \brief Magic number for NDArray list file  */
const uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief parse binary and load data to tensor (only load the data)
 * @param tensor the init tensor with no data
 * @param blob the binary
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadDataFromBinary(DLTensor *tensor, const char **blob) {
    uint64_t header;
    memcpy(&header, *blob, sizeof(header));
    *blob += sizeof(header);
    if (unlikely(header != kTVMNDArrayMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid DLTensor magic number: %" PRIX64 ", expect %" PRIX64,
                                header, kTVMNDArrayMagic);
    }
    *blob += sizeof(uint64_t);                                 // reserved
    *blob += sizeof(DLDevice);                                 // DLDevice

    if (unlikely(memcmp(&tensor->ndim, *blob, sizeof(int)))) { // ndim
        TVM_RT_SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, but got %d",
                                tensor->ndim, *(int *)(*blob));
    }
    *blob += sizeof(int); // ndim

    //    if (unlikely(memcmp(&tensor->dtype, *blob, sizeof(DLDataType)))) { // DLDateType
    //    }
    *blob += sizeof(DLDataType);             // DLDataType

    for (int i = 0; i < tensor->ndim; ++i) { // shapes
        const int64_t shape_i = *(int64_t *)(*blob);
        if (unlikely(tensor->shape[i] != shape_i)) {
            TVM_RT_SET_ERROR_RETURN(
                -1, "Invalid DLTensor shape: expect shape[%d] = %" PRIi64 ", but got %" PRIi64, i,
                tensor->shape[i], shape_i);
        }
        *blob += sizeof(int64_t); // shape
    }

    int64_t byte_size;
    memcpy(&byte_size, *blob, sizeof(byte_size));
    int64_t tensor_size = (int64_t)TVM_RT_WASM_DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "Invalid DLTensor byte size: expect %" PRIu64 ", but got %" PRIu64,
                                tensor_size, byte_size);
    }
    *blob += sizeof(byte_size); // byte_size

    if (tensor->device.device_type == kDLCPU || tensor->device.device_type == kDLCUDAHost) {
        memcpy(tensor->data, *blob, byte_size);
        *blob += byte_size;
        return 0;
    }

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

/*!
 * \brief parse binary and load data to tensor (only load the data)
 * @param tensor the init tensor with no data
 * @param fp the opened file struct
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadDataFromFile(DLTensor *tensor, FILE *fp) {
#define read_from_fp(ptr, len, fp)                                                                 \
    do {                                                                                           \
        if (unlikely(fread((ptr), 1, (len), fp) != (len))) {                                       \
            TVM_RT_SET_ERROR_RETURN(-1, "invalid param binary: unexpect EOF");                     \
        }                                                                                          \
    } while (0)

    uint64_t header;
    read_from_fp(&header, sizeof(uint64_t), fp);

    if (unlikely(header != kTVMNDArrayMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid DLTensor magic number: %" PRIX64 ", expect %" PRIX64,
                                header, kTVMNDArrayMagic);
    }

    read_from_fp(&header, sizeof(uint64_t), fp); // reserved
    DLDevice _d;
    read_from_fp(&_d, sizeof(DLDevice), fp);     // DLDevice
    (void)_d;

    int ndim;
    read_from_fp(&ndim, sizeof(int), fp); // ndim
    if (unlikely(tensor->ndim != ndim)) { // ndim
        TVM_RT_SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, but got %d",
                                tensor->ndim, ndim);
    }

    DLDataType _dlDataType;
    read_from_fp(&_dlDataType, sizeof(DLDataType), fp); // DLDataType
    (void)_dlDataType;

    for (int i = 0; i < tensor->ndim; ++i) { // shapes
        int64_t shape;
        read_from_fp(&shape, sizeof(int64_t), fp);
        if (unlikely(tensor->shape[i] != shape)) {
            TVM_RT_SET_ERROR_RETURN(
                -1, "Invalid DLTensor shape: expect shape[%d] = %" PRIi64 ", but got %" PRIi64, i,
                tensor->shape[i], shape);
        }
    }

    uint64_t byte_size;
    read_from_fp(&byte_size, sizeof(int64_t), fp);
    uint64_t tensor_size = TVM_RT_WASM_DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "Invalid DLTensor byte size: expect %" PRIu64 ", but got %" PRIu64,
                                tensor_size, byte_size);
    }

    if (tensor->device.device_type == kDLCPU || tensor->device.device_type == kDLCUDAHost) {
        read_from_fp(tensor->data, byte_size, fp);
        return 0;
    }

    void *buf = TVM_RT_WASM_WorkplaceMemoryAlloc(byte_size);

    size_t read_size = fread(buf, 1, byte_size, fp);
    if (read_size != byte_size) {
        TVM_RT_WASM_WorkplaceMemoryFree(buf);
        TVM_RT_SET_ERROR_RETURN(-1, "invalid param binary: unexpect EOF");
    }

    DLDevice cpu = {kDLCPU, 0};
    DLTensor src_tensor = {
        .ndim = tensor->ndim,
        .shape = tensor->shape,
        .dtype = tensor->dtype,
        .device = cpu,
        .data = buf,
    };

    // do copy data
    int status = TVMDeviceCopyDataFromTo(&src_tensor, tensor, NULL);

    TVM_RT_WASM_WorkplaceMemoryFree(buf);
    return status;
}
