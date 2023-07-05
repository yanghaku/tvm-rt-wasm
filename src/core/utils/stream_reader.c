/**
 * @file utils/stream_reader.c
 * @brief The utils functions to read object from bytes or file.
 */

#include <stdio.h>
#include <string.h>

#include <device/cpu_memory.h>
#include <utils/common.h>
#include <utils/stream_reader.h>

#define READ_BYTES_ERROR(error_code, expected_len, len)                                            \
    TVM_RT_SET_ERROR_RETURN(error_code, "Read bytes expect %zu but got %zu", expected_len, len)

#define SKIP_BYTES_ERROR(error_code, skip_len)                                                     \
    TVM_RT_SET_ERROR_RETURN(error_code, "Skip %zu bytes error.", skip_len)

typedef struct FileStreamReader {
    STREAM_READER_INTERFACE

    FILE *fp;
    char *temp_buffer;
    size_t temp_buffer_size;
} FileStreamReader;

static int TVM_RT_WASM_FileStreamReadBytes(struct StreamReader *r, void *buf, size_t expected_len) {
    FileStreamReader *reader = (FileStreamReader *)r;
    size_t len = fread(buf, 1, expected_len, reader->fp);
    if (unlikely(len != expected_len)) {
        READ_BYTES_ERROR(-1, expected_len, len);
    }
    return 0;
}

static int TVM_RT_WASM_FileStreamSkipBytes(struct StreamReader *r, size_t skip_len) {
    FileStreamReader *reader = (FileStreamReader *)r;
    int status = fseek(reader->fp, (long)skip_len, SEEK_CUR);
    if (unlikely(status)) {
        SKIP_BYTES_ERROR(status, skip_len);
    }
    return 0;
}

static int TVM_RT_WASM_FileStreamFree(struct StreamReader *r) {
    int status = 0;
    FileStreamReader *reader = (FileStreamReader *)r;
    if (reader->fp) {
        status = fclose(reader->fp);
    }
    if (reader->temp_buffer) {
        TVM_RT_WASM_HeapMemoryFree(reader->temp_buffer);
    }
    TVM_RT_WASM_HeapMemoryFree(reader);
    return status;
}

#define MIN_BUFFER_SIZE 512
static const char *TVM_RT_WASM_FileStreamReadToBuffer(struct StreamReader *r, size_t expected_len) {
    FileStreamReader *reader = (FileStreamReader *)r;

    // need realloc
    if (expected_len > reader->temp_buffer_size) {
        reader->temp_buffer_size = MAX(expected_len, MIN_BUFFER_SIZE);
        if (reader->temp_buffer) {
            TVM_RT_WASM_HeapMemoryFree(reader->temp_buffer);
        }
        reader->temp_buffer = TVM_RT_WASM_HeapMemoryAlloc(reader->temp_buffer_size);
    }

    size_t len = fread(reader->temp_buffer, 1, expected_len, reader->fp);
    if (unlikely(len != expected_len)) {
        READ_BYTES_ERROR(NULL, expected_len, len);
    }
    return reader->temp_buffer;
}

int TVM_RT_WASM_FileStreamReaderCreate(const char *filename, StreamReader **reader_ptr) {
    *reader_ptr = NULL;
    CHECK_INPUT_POINTER(filename, -2, "Filename");

    // todo: if mmap can be used, use it.
    FILE *fp = fopen(filename, "rb");
    if (unlikely(!fp)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot open file `%s`", filename);
    }

    FileStreamReader *reader = TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct FileStreamReader));
    reader->fp = fp;
    reader->temp_buffer = NULL;
    reader->temp_buffer_size = 0;
    reader->ReadBytes = TVM_RT_WASM_FileStreamReadBytes;
    reader->SkipBytes = TVM_RT_WASM_FileStreamSkipBytes;
    reader->Free = TVM_RT_WASM_FileStreamFree;
    reader->ReadToBuffer = TVM_RT_WASM_FileStreamReadToBuffer;

    *reader_ptr = (StreamReader *)reader;
    return 0;
}

typedef struct BytesStreamReader {
    STREAM_READER_INTERFACE

    const char *bytes;
    const char *bytes_end;
} BytesStreamReader;

static int TVM_RT_WASM_BytesStreamReadBytes(struct StreamReader *r, void *buf,
                                            size_t expected_len) {
    BytesStreamReader *reader = (BytesStreamReader *)r;
    const char *next = reader->bytes + expected_len;
    if (next >= reader->bytes_end) {
        READ_BYTES_ERROR(-1, expected_len, reader->bytes_end - reader->bytes);
    }
    memcpy(buf, reader->bytes, expected_len);
    reader->bytes = next;
    return 0;
}

static int TVM_RT_WASM_BytesStreamSkipBytes(struct StreamReader *r, size_t skip_len) {
    BytesStreamReader *reader = (BytesStreamReader *)r;
    const char *next = reader->bytes + skip_len;
    if (next >= reader->bytes_end) {
        SKIP_BYTES_ERROR(-1, skip_len);
    }
    reader->bytes = next;
    return 0;
}

static int TVM_RT_WASM_BytesStreamFree(struct StreamReader *reader) {
    // do nothing
    (void)reader;
    return 0;
}

static const char *TVM_RT_WASM_BytesStreamReadToBuffer(struct StreamReader *r,
                                                       size_t expected_len) {
    BytesStreamReader *reader = (BytesStreamReader *)r;
    const char *res = reader->bytes;
    const char *next = res + expected_len;
    if (next >= reader->bytes_end) {
        READ_BYTES_ERROR(NULL, expected_len, reader->bytes_end - res);
    }
    reader->bytes = next;
    return res;
}

int TVM_RT_WASM_BytesStreamReaderCreate(const char *bytes, size_t byte_len,
                                        StreamReader **reader_ptr) {
    CHECK_INPUT_POINTER(bytes, -2, "Bytes");
    *reader_ptr = NULL;

    // check overflow
    uintptr_t b = (uintptr_t)bytes;
    if (UINTPTR_MAX - byte_len < b) {
        TVM_RT_SET_ERROR_RETURN(-1, "Bytes length overflow!");
    }

    const char *bytes_end = bytes + byte_len;
    BytesStreamReader *reader = TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct BytesStreamReader));
    reader->bytes = bytes;
    reader->bytes_end = bytes_end;
    reader->ReadBytes = TVM_RT_WASM_BytesStreamReadBytes;
    reader->SkipBytes = TVM_RT_WASM_BytesStreamSkipBytes;
    reader->Free = TVM_RT_WASM_BytesStreamFree;
    reader->ReadToBuffer = TVM_RT_WASM_BytesStreamReadToBuffer;

    *reader_ptr = (StreamReader *)reader;
    return 0;
}
