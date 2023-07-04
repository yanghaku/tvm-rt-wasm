/*!
 * @file utils/stream_reader.h
 * @brief the utils function to read object from bytes/file.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#ifndef TVM_RT_WASM_CORE_UTILS_STREAM_READER_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_STREAM_READER_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define STREAM_READER_INTERFACE                                                                    \
    /**                                                                                            \
     * @brief Read the bytes to buffer, if eof, it will call TVMSetLastError.                      \
     * @param reader The reader instance.                                                          \
     * @param buf The buffer to receive bytes.                                                     \
     * @param expected_len The expected byte length.                                               \
     * @return 0 if successful.                                                                    \
     */                                                                                            \
    int (*ReadBytes)(struct StreamReader * reader, void *buf, size_t expected_len);                \
                                                                                                   \
    /**                                                                                            \
     * @brief Skip the bytes, if eof, it will call TVMSetLastError.                                \
     * @param reader The reader instance.                                                          \
     * @param skip_len The length to skip.                                                         \
     * @return 0 if successful.                                                                    \
     */                                                                                            \
    int (*SkipBytes)(struct StreamReader * reader, size_t skip_len);                               \
                                                                                                   \
    /**                                                                                            \
     * @brief Free the reader.                                                                     \
     * @param reader The reader instance.                                                          \
     * @return 0 if successful.                                                                    \
     */                                                                                            \
    int (*Free)(struct StreamReader * reader);                                                     \
                                                                                                   \
    /**                                                                                            \
     * @brief Read the bytes to inner temporary buffer.                                            \
     * @note The buffer can only be used before next call.                                         \
     * @param expected_len the length to read.                                                     \
     * @return The temporary buffer, NULL if fail.                                                 \
     */                                                                                            \
    const char *(*ReadToBuffer)(struct StreamReader * reader, size_t expected_len);

/** @brief The stream reader, read bytes from byte or file stream. */
typedef struct StreamReader {
    STREAM_READER_INTERFACE
} StreamReader;

/**
 * @brief Create a new FileStreamReader instance.
 * @param filename The filename.
 * @param reader_ptr The pointer to receive instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_FileStreamReaderCreate(const char *filename, StreamReader **reader_ptr);

/**
 * @brief Create a new BytesStreamReader instance.
 * @param bytes The byte array.
 * @param byte_len The byte array length.
 * @param reader_ptr The pointer to receive instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_BytesStreamReaderCreate(const char *bytes, size_t byte_len,
                                        StreamReader **reader_ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_STREAM_READER_H_INCLUDE_
