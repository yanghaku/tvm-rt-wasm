/**
 * @file utils/binary_reader.h
 * @brief The utils functions to read object from binary blob.
 */

#ifndef TVM_RT_WASM_CORE_UTILS_BINARY_READER_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_BINARY_READER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <utils/common.h>

/** @brief Binary blob reader
 * The different between StreamReader and BinaryReader:
 * StreamReader: The data will be freed after reader process.
 * BinaryReader: The data will **not be freed**.
 */
typedef struct {
    const char *current_ptr;
    const char *const end_ptr;
} BinaryReader;

/**
 * @brief Add read_size offset to current pointer and check.
 * If check fail, goto fail label.
 * Requirement variables:
 *  int status;
 *  BinaryReader *reader;
 */
#define TVM_RT_WASM_BinaryCheckReadOrGoto(_ptr, _read_size, _fail_label)                           \
    do {                                                                                           \
        (_ptr) = reader->current_ptr;                                                              \
        const char *_next = (_ptr) + (_read_size);                                                 \
        if (unlikely(_next > reader->end_ptr)) {                                                   \
            status = -1;                                                                           \
            TVM_RT_SET_ERROR_AND_GOTO(_fail_label, "Module binary unexpected eof.");               \
        }                                                                                          \
        reader->current_ptr = _next;                                                               \
    } while (0)

/**
 * @brief Create a new BinaryReader.
 * If the blob is invalid, current_ptr is NULL. The error message is in tvm last error.
 * @param blob The binary blob.
 * @param blob_size The byte number of blob.
 * @return The created instance. Fail if current_ptr is NULL.
 */
INLINE BinaryReader TVM_RT_WASM_BinaryReaderCreate(const char *blob, size_t blob_size) {
    BinaryReader reader = {
        .current_ptr = blob,
        .end_ptr = blob + blob_size,
    };
    CHECK_INPUT_POINTER(blob, reader, "Bytes");
    // check overflow and create binary reader
    uintptr_t b = (uintptr_t)blob;
    if (UINTPTR_MAX - blob_size < b) {
        blob = NULL;
        TVM_RT_SET_ERROR("Bytes length overflow!");
    }
    return reader;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_BINARY_READER_H_INCLUDE_
