/*!
 * \file graph/json.c
 * \brief the implementation for json api.
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <graph/json.h>

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
        if (likely(status > 0)) {      // read key success
            NextNonSpace(*reader, ch); // read the next ':'
            CheckEQ(ch, ':');          // if not ':', return error code
        }
        return status;
    } else if (ch == '}') {
        return 0;
    }
    return -2;
}
