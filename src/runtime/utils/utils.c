/*!
 * \file src/runtime/module/module.c
 * \brief in utils/ *.h, some function cannot be inlined, so implement in this file
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/runtime/utils/json.h>
#include <tvm/runtime/utils/tensor_helper.h>
#include <tvm/runtime/utils/trie.h>

/*!----------------------------------------for utils/json.h ----------------------------------------------------------*/

/*! \brief this function contain function call, so cannot be inlined */
int TVM_RT_WASM_JsonReader_NextObjectItem(JsonReader *reader, char *out_key, size_t out_key_size) {
    char ch;
    NextNonSpace(*reader, ch);
    if (likely(ch == '{' || ch == ',')) {
        PeekNextNonSpace(*reader, ch);
        if (unlikely(ch == '}')) {           // the end of object
            return NextChar(*reader) != '}'; // read this '}'
        }
        int status = TVM_RT_WASM_JsonReader_ReadString(reader, out_key, out_key_size); // read key
        if (likely(status > 0)) {                                                      // read key success
            NextNonSpace(*reader, ch);                                                 // read the next ':'
            CheckEQ(ch, ':');                                                          // if not ':', return error code
        }
        return status;
    } else if (ch == '}') {
        return 0;
    }
    return -2;
}

/*!----------------------------------------for utils/tensor-helper.h -------------------------------------------------*/

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
        SET_ERROR_RETURN(-1, "Invalid DLTensor file Magic number: %llX, expect %llX\n", header, kTVMNDArrayMagic);
    }
    *blob += sizeof(uint64_t); // reserved
    *blob += sizeof(DLDevice); // DLDevice

    if (unlikely(memcmp(&tensor->ndim, *blob, sizeof(int)))) { // ndim
        SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, given %d", tensor->ndim, *(int *)(*blob));
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
    int64_t tensor_size = (int64_t)TVM_RT_WASM_DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        SET_ERROR_RETURN(-1, "Invalid DLTensor ata byte size: expect %llu, but given %llu\n", tensor_size, byte_size);
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
#define read_from_fp(ptr, len, fp)                                                                                     \
    do {                                                                                                               \
        if (unlikely(fread((ptr), 1, (len), fp) != (len))) {                                                           \
                                                                                                                       \
            SET_ERROR_RETURN(-1, "invalid param binary: unexpect EOF");                                                \
        }                                                                                                              \
    } while (0)

    uint64_t header;
    read_from_fp(&header, sizeof(uint64_t), fp);

    if (unlikely(header != kTVMNDArrayMagic)) {
        SET_ERROR_RETURN(-1, "Invalid DLTensor file Magic number: %llX, expect %llX\n", header, kTVMNDArrayMagic);
    }

    read_from_fp(&header, sizeof(uint64_t), fp); // reserved
    DLDevice _d;
    read_from_fp(&_d, sizeof(DLDevice), fp); // DLDevice
    (void)_d;

    int ndim;
    read_from_fp(&ndim, sizeof(int), fp); // ndim
    if (unlikely(tensor->ndim != ndim)) { // ndim
        SET_ERROR_RETURN(-1, "DLTensor ndim must be same: expected %d, given %d", tensor->ndim, ndim);
    }

    DLDataType _dlDataType;
    read_from_fp(&_dlDataType, sizeof(DLDataType), fp); // DLDataType
    (void)_dlDataType;

    for (int i = 0; i < tensor->ndim; ++i) { // shapes
        uint64_t shape;
        read_from_fp(&shape, sizeof(uint64_t), fp);
        if (unlikely(tensor->shape[i] != shape)) {
            SET_ERROR_RETURN(-1, "Invalid DLTensor shape: expect shape[%d] = %lld, but given %lld\n", i,
                             tensor->shape[i], shape);
        }
    }

    int64_t byte_size;
    read_from_fp(&byte_size, sizeof(int64_t), fp);
    int64_t tensor_size = (int64_t)TVM_RT_WASM_DLTensor_GetDataBytes(tensor);
    if (unlikely(byte_size != tensor_size)) {
        SET_ERROR_RETURN(-1, "Invalid DLTensor ata byte size: expect %llu, but given %llu\n", tensor_size, byte_size);
    }

    if (tensor->device.device_type == kDLCPU || tensor->device.device_type == kDLCUDAHost) {
        read_from_fp(tensor->data, byte_size, fp);
        return 0;
    }

    void *buf = TVM_RT_WASM_WorkplaceMemoryAlloc(byte_size);

    size_t read_size = fread(buf, 1, byte_size, fp);
    if (read_size != byte_size) {
        TVM_RT_WASM_WorkplaceMemoryFree(buf);
        SET_ERROR_RETURN(-1, "invalid param binary: unexpect EOF");
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

/*!----------------------------------------for utils/trie.h ----------------------------------------------------------*/

/*! \brief this is a table for char to index (for all uint8_t )  for Trie */
const unsigned char char2index[] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 0,   255, 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  255, 255, 255, 255, 255, 255, 255, 11,
    12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
    34,  35,  36,  255, 255, 255, 255, 37,  255, 38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,
    51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

const char index2char[] = {'.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                           'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                           'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

/*! \brief These functions are recursive and cannot be inlined */

int TVM_RT_WASM_TrieInsertAll(Trie *dst, Trie *src) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (src->son[i]) {
            if (dst->son[i] == NULL) {
                dst->son[i] = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Trie));
                memset(dst->son[i], 0, sizeof(Trie));
            }
            TVM_RT_WASM_TrieInsertAll(dst->son[i], src->son[i]);
        }
    }
    if (dst->data == NULL) {
        dst->data = src->data;
    }
    return 0;
}

void TVM_RT_WASM_TrieVisit(Trie *trie, void (*visit)(char, void **, void *), void *source_handle) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            if (trie->son[i]->data) {
                visit(index2char[i], &trie->son[i]->data, source_handle);
            }
            TVM_RT_WASM_TrieVisit(trie->son[i], visit, source_handle);
        }
    }
}

int TVM_RT_WASM_TrieClone(const Trie *trie, Trie **cloned) {
    *cloned = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Trie));
    memcpy(*cloned, trie, sizeof(Trie));
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TVM_RT_WASM_TrieClone(trie->son[i], &(*cloned)->son[i]);
        }
    }
    return 0;
}

void TVM_RT_WASM_TrieRelease(Trie *trie) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TVM_RT_WASM_TrieRelease(trie->son[i]);
        }
    }
    TVM_RT_WASM_HeapMemoryFree(trie);
}
