/**
 * @file utils/tensor_helper.h
 * @brief the utils function for DLTensor
 */

#include <string.h>

#include <device/cpu_memory.h>
#include <device/device_api.h>
#include <utils/common.h>
#include <utils/tensor_helper.h>

int TVM_RT_WASM_DLTensor_LoadFromStream(DLTensor *tensor, StreamReader *reader) {
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
        tensor->strides = NULL;
        tensor->byte_offset = 0;
    }

    // get and check byte size
    uint64_t byte_size;
    status = reader->ReadBytes(reader, &byte_size, sizeof(int64_t));
    if (unlikely(status)) {
        return status;
    }
    size_t tensor_byte_size =
        TVM_RT_WASM_DLTensor_GetDataBytes(tensor->shape, tensor->ndim, tensor->dtype);
    if (unlikely(byte_size != (uint64_t)tensor_byte_size)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid DLTensor byte size: expect %zu, but got %" PRIu64,
                                tensor_byte_size, byte_size);
    }

    if (tensor->data == NULL) {
        tensor->data = TVM_RT_WASM_HeapMemoryAlignedAlloc(tensor_byte_size);
    }

    if (tensor->device.device_type == kDLCPU || tensor->device.device_type == kDLCUDAHost) {
        return reader->ReadBytes(reader, tensor->data, tensor_byte_size);
    }

    const char *data_buf = reader->ReadToBuffer(reader, tensor_byte_size);
    if (unlikely(data_buf == NULL)) {
        return -1;
    }

    DeviceAPI *device_api;
    status = TVM_RT_WASM_DeviceAPIGet(tensor->device.device_type, &device_api);
    if (unlikely(status)) {
        return status;
    }
    return device_api->CopyDataFromCPUToDevice(data_buf, tensor->data, tensor_byte_size, 0,
                                               tensor->byte_offset, NULL, tensor->device.device_id);
}

int TVM_RT_WASM_DLTensor_LoadFromBinary(DLTensor *tensor, BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    // head magic
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    uint64_t header = *(uint64_t *)cur_ptr;
    if (unlikely(header != kTVMNDArrayMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid DLTensor magic number: %" PRIX64 ", expect %" PRIX64,
                                header, kTVMNDArrayMagic);
    }

    // reserved
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);

    // DLDevice
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(DLDevice), load_fail);
    tensor->device = *(const DLDevice *)cur_ptr;
    if (unlikely(tensor->device.device_type != kDLCPU)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_fail, "Invalid DLTensor device must be CPU");
    }

    // ndim
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int), load_fail);
    tensor->ndim = *(const int *)cur_ptr;

    // DLDataType
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(DLDataType), load_fail);
    tensor->dtype = *(const DLDataType *)cur_ptr;

    // Shape
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * tensor->ndim, load_fail);
    tensor->shape = (int64_t *)cur_ptr;

    // Get and check byte size
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    const uint64_t byte_size = *(const uint64_t *)cur_ptr;
    const uint64_t check_byte_size =
        (uint64_t)TVM_RT_WASM_DLTensor_GetDataBytes(tensor->shape, tensor->ndim, tensor->dtype);
    if (unlikely(byte_size != check_byte_size)) {
        TVM_RT_SET_ERROR_RETURN(-1,
                                "Invalid DLTensor byte size: expect %" PRIu64 ", but got %" PRIu64,
                                check_byte_size, byte_size);
    }

    // Data bytes
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, (size_t)byte_size, load_fail);
    tensor->data = (void *)cur_ptr;
    tensor->strides = NULL;
    tensor->byte_offset = 0;

load_fail:
    return status;
}
