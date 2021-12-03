/*!
 * \file runtime/utils/json.h
 * \brief the parse util for load graph json
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_JSON_H
#define TVM_RT_JSON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/utils/common.h>

/*! \brief get next char from the str, and change pointer to next */
#define NextChar(ptr) (*((ptr)++))

/*! \brief get next char from the str, but not change pointer */
#define PeekNextChar(ptr) (*(ptr))

/*! \brief get next nonSpace char, and save it to "ch" , pointer will point to the next of "ch" */
#define NextNonSpace(ptr, ch)                                                                                          \
    do {                                                                                                               \
        (ch) = NextChar(ptr);                                                                                          \
    } while (isspace(ch))

/*! \brief get next nonSpace char, and save it to "ch" , pointer will point to "ch" */
#define PeekNextNonSpace(ptr, ch)                                                                                      \
    do {                                                                                                               \
        while (isspace(PeekNextChar(ptr)))                                                                             \
            ++(ptr);                                                                                                   \
        (ch) = *(ptr);                                                                                                 \
    } while (0)

/*! \brief check the expect char and given char equal if or not */
#define CheckEQ(expect, given)                                                                                         \
    do {                                                                                                               \
        if (unlikely((expect) != (given))) {                                                                           \
            fprintf(stderr, "json parse error: expect char is '%c'(ascii=%d), but given '%c'(ascii=%d)\n", (expect),   \
                    (expect), (given), (given));                                                                       \
            return -2;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define isdigit0to9(ch) ((ch) >= '0' && (ch) <= '9')
#define isdigit1to9(ch) ((ch) >= '1' && (ch) <= '9')

/*! \brief the loop read number from string that the ptr point to */
// todo: change it to judge lead zero
#define str2numLoop(ptr, num)                                                                                          \
    do {                                                                                                               \
        while (isdigit0to9(PeekNextChar(ptr))) {                                                                       \
            (num) = ((num) << 3) + ((num) << 1) + NextChar(ptr) - '0';                                                 \
        }                                                                                                              \
    } while (0)

/*! \brief change string to unsigned(any bits) and save to num */
#define str2unsigned(ptr, num)                                                                                         \
    do {                                                                                                               \
        char ch;                                                                                                       \
        NextNonSpace(ptr, ch);                                                                                         \
        if (likely(isdigit0to9(ch))) {                                                                                 \
            (num) = ch - '0';                                                                                          \
            str2numLoop(ptr, (num));                                                                                   \
            return 0;                                                                                                  \
        }                                                                                                              \
        return -1;                                                                                                     \
    } while (0)

/*! \brief change string to signed(any bits) and save to num */
#define str2signed(ptr, num)                                                                                           \
    do {                                                                                                               \
        char ch;                                                                                                       \
        /* if negative, flag=1, else 0*/                                                                               \
        char flag = 0;                                                                                                 \
        NextNonSpace(ptr, ch);                                                                                         \
        if (ch == '-') {                                                                                               \
            flag = 1;                                                                                                  \
            ch = NextChar(ptr);                                                                                        \
        }                                                                                                              \
        if (likely(isdigit0to9(ch))) {                                                                                 \
            (num) = ch - '0';                                                                                          \
            str2numLoop(ptr, (num));                                                                                   \
            if (flag) {                                                                                                \
                (num) = -(num);                                                                                        \
            }                                                                                                          \
            return 0;                                                                                                  \
        }                                                                                                              \
        return -1;                                                                                                     \
    } while (0)

/*-----------------------------------Definition for JsonReader--------------------------------------------------------*/

/*! \brief a tiny json reader simply contain a char pointer */
typedef const char *JsonReader;

/*!
 * \brief Constructor function for JsonReader
 * @param json_str the source json string pointer
 * @param out_reader the pointer to receive out_reader
 * @return 0 if successful
 */
INLINE int JsonReader_Create(const char *json_str, JsonReader **out_reader) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    int status = TVMDeviceAllocDataSpace(cpu, sizeof(JsonReader), 0, no_type, (void **)&out_reader);
    **out_reader = json_str;
    return status;
}

/*!
 * \brief delete the instance of JsonReader
 * @param reader the instance pointer
 * @return 0 if successful
 */
INLINE int JsonReader_Release(JsonReader *reader) {
    DLDevice cpu = {kDLCPU, 0};
    return TVMDeviceFreeDataSpace(cpu, reader);
}

/*!
 * \brief read a 32 bit unsigned int
 * \note The reason for using inline instead of macro definition is the need for type checking
 * @param reader the instance of JsonReader
 * @param out_num the pointer to receive number
 * @return 0 if successful
 */
INLINE int JsonReader_Read_uint32(JsonReader *reader, uint32_t *out_num) { str2unsigned(*reader, *out_num); }

/*!
 * \brief read a 32 bit signed int
 * @param reader the instance of JsonReader
 * @param out_num the pointer to receive number
 * @return 0 if successful
 */
INLINE int JsonReader_Read_int32(JsonReader *reader, int32_t *out_num) { str2signed(*reader, *out_num); }

/*!
 * \brief read a 64 bit unsigned int
 * @param reader the instance of JsonReader
 * @param out_num the pointer to receive number
 * @return 0 if successful
 */
INLINE int JsonReader_Read_uint64(JsonReader *reader, uint64_t *out_num) { str2unsigned(*reader, *out_num); }

/*!
 * \brief read a 64 bit signed int
 * @param reader the instance of JsonReader
 * @param out_num the pointer to receive number
 * @return 0 if successful
 */
INLINE int JsonReader_Read_int64(JsonReader *reader, int64_t *out_num) { str2signed(*reader, *out_num); }

/*!
 * \brief read string and save to out_str
 * @param reader the instance of JsonReader
 * @param out_str the store buffer
 * @param out_str_size the store buffer size
 * @return if successful return actual length ( > 0 ), -1: buffer_size tool short, <= -1: error code
 */
INLINE int JsonReader_ReadString(JsonReader *reader, char *out_str, size_t out_str_size) {
    char ch;
    NextNonSpace(*reader, ch);
    CheckEQ(ch, '\"'); // check the start is '\"'
    int size = 0;
    while ((ch = NextChar(*reader))) {
        if (unlikely(ch == '\"')) { // end of string
            out_str[size] = '\0';
            if (likely(size > 0)) {
                return size;
            }
            return -2; // empty string
        }

        if (unlikely(ch == '\\')) { // escape character
            ch = NextChar(*reader);
            switch (ch) {
            case 'n':
                ch = '\n';
                break;
            case 'r':
                ch = '\r';
                break;
            case '\"':
                ch = '\"';
                break;
            case '\\':
                ch = '\\';
                break;
            case 't':
                ch = '\t';
                break;
            case 'b':
                ch = '\b';
                break;
            default:
                fprintf(stderr, "Error: unsupported string escape %c(ascii=%d)", ch, ch);
                return -2;
            }
        }

        int new_size = size + 1;
        if (unlikely(new_size == (int)out_str_size)) { // buf is two short
            fprintf(stderr, "Error: string buf is too short! now buf size = %d\n", new_size);
            return -1;
        }
        out_str[size] = ch;
        size = new_size;
    }
    return size;
}

/*!
 * \brief check and prepare to read the next array item
 * @param reader the instance of JsonReader
 * @return if successful return 1, 0: no array item, <0 : error code
 */
INLINE int JsonReader_NextArrayItem(JsonReader *reader) {
    char ch;
    NextNonSpace(*reader, ch);
    if (likely(ch == '[' || ch == ',')) {
        PeekNextNonSpace(*reader, ch);
        if (unlikely(ch == ']')) { // the end of array
            NextChar(*reader);     // read this ']'
            return 0;
        }
        return 1;
    } else if (ch == ']') {
        return 0;
    }
    return -1;
}

/*!
 * \brief prepare to read the next object item value, and read it's key if exist
 * @param reader the instance of JsonReader
 * @param out_key the buffer to save "key" of object
 * @param out_key_size buffer size
 * @return if successful return actual length of key ( >0 ), 0: no object to read, -1: buffer_size tool short, <= -2:
 * error
 */
INLINE int JsonReader_NextObjectItem(JsonReader *reader, char *out_key, size_t out_key_size) {
    char ch;
    NextNonSpace(*reader, ch);
    if (likely(ch == '{' || ch == ',')) {
        PeekNextNonSpace(*reader, ch);
        if (unlikely(ch == '}')) { // the end of object
            NextChar(reader);      // read this '}'
            return 0;
        }
        int status = JsonReader_ReadString(reader, out_key, out_key_size); // read key
        if (likely(status > 0)) {                                          // read key success
            NextNonSpace(*reader, ch);                                     // read the next ':'
            CheckEQ(ch, ':');                                              // if not ':', return error code
        }
        return status;
    } else if (ch == '}') {
        return 0;
    }
    return -2;
}

/*!
 * \brief get the array length (it need to scan all the json string of array)
 * @param reader the instance of JsonReader
 * @param out_size the pointer to receive array length
 * @return 0 if successful
 */
INLINE int JsonReader_ArrayLength(JsonReader *reader, size_t *out_size) {
    const char *ptr = *reader;
    int now_dep = 0;
    int now_size = 0;
    char ch;

    NextNonSpace(ptr, ch); // read the start of ptr
    CheckEQ(ch, '[');      // check it is '['

    ch = NextChar(ptr);
    while (1) {
        switch (ch) {

        case '\0': {
            return -1;
        }
        case '\"': {
            do {
                ch = NextChar(ptr);
                if (unlikely(ch == '\\')) { // escape character
                    ch = NextChar(ptr);     // discard it
                    ch = NextChar(ptr);
                }
            } while (ch && ch != '\"');
            break;
        }
        case '[':
        case '{':
            ++now_dep;
            break;
        case '}': {
            --now_dep;
            break;
        }
        case ']': {
            if (now_dep == 0) {
                *out_size = now_size + 1;
                return 0;
            }
            --now_dep;
            break;
        }
        case ',': {
            if (now_dep == 0) {
                ++now_size;
            }
            break;
        }
        default:
            break;
        }

        ch = NextChar(ptr);
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_JSON_H
