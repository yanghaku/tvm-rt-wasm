/*!
 * @file utils/tensor_helper.h
 * @brief the utils function for DLTensor
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

/*!
 * @brief Parse binary and load data to tensor.
 * @param tensor the tensor with no data or with data.
 * @param reader The stream reader instance.
 * @note If shape is NULL, it will alloc memory for shape.
 * @note If data is NULL, it will alloc memory for data.
 * @return 0 if successful
 */
int TVM_RT_WASM_DLTensor_LoadFromReader(DLTensor *tensor, StreamReader *reader) {
    uint64_t header;
    int status = reader->ReadBytes(reader, &header, sizeof(uint64_t));
    if (unlikely(status)) {
        return status;
    }
    if (unlikely(header != kTVMNDArrayMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid DLTensor magic number: %" PRIX64 ", expect %" PRIX64,
                                header, kTVMNDArrayMagic);
    }

    status = reader->SkipBytes(reader, sizeof(uint64_t) + sizeof(DLDevice)); // reserved + DLDevice
    if (unlikely(status)) {
        return status;
    }

    int ndim;
    status = reader->ReadBytes(reader, &ndim, sizeof(int));
    if (unlikely(status)) {
        return status;
    }
    if (tensor->ndim != 0 && unlikely(tensor->ndim != ndim)) { // ndim
        TVM_RT_SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, but got %d",
                                tensor->ndim, ndim);
    }

    DLDataType dl_data_type;
    status = reader->ReadBytes(reader, &dl_data_type, sizeof(DLDataType)); // DLDataType
    if (unlikely(status)) {
        return status;
    }

    if (tensor->shape) {                 // tensor has been init.
        for (int i = 0; i < ndim; ++i) { // shapes
            int64_t shape;
            status = reader->ReadBytes(reader, &shape, sizeof(uint64_t));
            if (unlikely(status)) {
                return status;
            }
            if (unlikely(tensor->shape[i] != shape)) {
                TVM_RT_SET_ERROR_RETURN(
                    -1, "Invalid DLTensor shape: expect shape[%d] = %" PRIi64 ", but got %" PRIi64,
                    i, tensor->shape[i], shape);
            }
        }
    } else { // need to init tensor.
        tensor->ndim = ndim;
        tensor->dtype = dl_data_type;
        tensor->device.device_type = kDLCPU;
        tensor->shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * ndim);
        status = reader->ReadBytes(reader, tensor->shape, sizeof(int64_t) * ndim);
        if (unlikely(status)) {
            return status;
        }
    }

    // get and check byte size
    uint64_t byte_size;
    status = reader->ReadBytes(reader, &byte_size, sizeof(int64_t));
    if (unlikely(status)) {
        return status;
    }
    uint64_t tensor_size = TVM_RT_WASM_DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "Invalid DLTensor byte size: expect %" PRIu64 ", but got %" PRIu64,
                                tensor_size, byte_size);
    }

    if (tensor->data == NULL) {
        tensor->data = TVM_RT_WASM_HeapMemoryAlloc(byte_size);
    }

    if (tensor->device.device_type == kDLCPU || tensor->device.device_type == kDLCUDAHost) {
        return reader->ReadBytes(reader, tensor->data, byte_size);
    }

    const char *data_buf = reader->ReadToBuffer(reader, byte_size);
    if (unlikely(data_buf == NULL)) {
        return -1;
    }

    DLDevice cpu = {kDLCPU, 0};
    DLTensor src_tensor = {
        .ndim = tensor->ndim,
        .shape = tensor->shape,
        .dtype = tensor->dtype,
        .device = cpu,
        .data = (void *)data_buf,
    };

    // do copy data
    return TVMDeviceCopyDataFromTo(&src_tensor, tensor, NULL);
}
