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
 * \file src/runtime/utils/trie.c
 * \brief the implement for trie
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/trie.h>

/*! \brief charset = 0-9,a-z,A-Z, _, : */
#define CHAR_SET_SIZE 64

/*! \brief this is a table for char to index */
static const unsigned char char2index[] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0,   255, 1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  255, 255, 255, 255, 255, 255, 255, 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  255, 255, 255, 255,
    37,  255, 38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
    55,  56,  57,  58,  59,  60,  61,  62,  63,  255, 255, 255, 255, 255};

static const char index2char[] = {'.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                                  'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a',
                                  'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

/*! \brief the definition of Trie */
typedef struct Trie {
    Trie *son[CHAR_SET_SIZE];
    void *data;
} Trie;

Trie *TrieCreate() {
    DLDevice cpu = {kDLCPU, 0};
    Trie *trie;
    memory_alloc(sizeof(Trie), cpu, (void **)&trie);
    memset(trie, 0, sizeof(Trie));
    return trie;
}

int TrieInsert(Trie *trie, const char *name, void *data) {

    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            return -1;
        }
        if (trie->son[id] == NULL) {
            trie->son[id] = TrieCreate();
        }
        trie = trie->son[id];
        ++name;
    }
    trie->data = data;
    return 0;
}

int TrieQuery(Trie *trie, const char *name, void **data) {

    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            return -1;
        }
        if (trie->son[id]) {
            trie = trie->son[id];
            ++name;
        } else {
            *data = NULL;
            return -2;
        }
    }
    *data = trie->data;
    return 0;
}

void TrieVisit(Trie *trie, void (*visit)(char c, void *data, void *source_handle),
               void *source_handle) {

    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            if (trie->son[i]->data) {
                visit(index2char[i], trie->son[i]->data, source_handle);
            }
            TrieVisit(trie->son[i], visit, source_handle);
        }
    }
}

int TrieRelease(Trie *trie) {
    DLDevice cpu = {kDLCPU, 0};
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TrieRelease(trie->son[i]);
        }
    }
    return memory_free(cpu, trie);
}
