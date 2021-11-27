/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/utils/json.c
 * \brief the implement for json reader
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/json.h>

/*!
 * \brief get the array length (it need to scan all the json string of array)
 * @param reader the instance of JsonReader
 * @param out_size the pointer to receive array length
 * @return 0 if successful
 */
int JsonReader_ArrayLength(JsonReader *reader, size_t *out_size) {
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
