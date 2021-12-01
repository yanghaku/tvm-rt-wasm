/*!
 * \file src/runtime/utils/trie.c
 * \brief the implement for trie
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/utils/common.h>
#include <tvm/runtime/utils/trie.h>

/*! \brief charset = 0-9,a-z,A-Z, _, : */
#define CHAR_SET_SIZE 64

/*! \brief this is a table for char to index (for all uint8_t ) */
static const unsigned char char2index[] = {
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

static const char index2char[] = {'.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
                                  'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                                  'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

/*! \brief the definition of Trie */
typedef struct Trie {
    Trie *son[CHAR_SET_SIZE];
    void *data;
} Trie;

int TrieCreate(Trie **trie) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    int status = TVMDeviceAllocDataSpace(cpu, sizeof(Trie), 0, no_type, (void **)&trie);
    memset(*trie, 0, sizeof(Trie));
    return status;
}

int TrieInsert(Trie *trie, const uint8_t *name, void *data) {
    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            SET_ERROR_RETURN(TRIE_INVALID_CHARSET, "charset is invalid: %c(ascii=%d)", *name, *name);
        }
        if (trie->son[id] == NULL) {
            TrieCreate(&trie->son[id]);
        }
        trie = trie->son[id];
        ++name;
    }
    trie->data = data;
    return TRIE_SUCCESS;
}

int TrieQuery(Trie *trie, const uint8_t *name, void **data) {
    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            SET_ERROR_RETURN(TRIE_INVALID_CHARSET, "charset is invalid: %c(ascii=%d)", *name, *name);
        }
        if (trie->son[id]) {
            trie = trie->son[id];
            ++name;
        } else {
            *data = NULL;
            return TRIE_NOT_FOUND;
        }
    }
    *data = trie->data;
    return TRIE_SUCCESS;
}

void TrieVisit(Trie *trie, void (*visit)(char, void *, void *), void *source_handle) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            if (trie->son[i]->data) {
                visit(index2char[i], trie->son[i]->data, source_handle);
            }
            TrieVisit(trie->son[i], visit, source_handle);
        }
    }
}

int TrieClone(const Trie *trie, Trie **cloned) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(Trie), 0, no_type, (void **)&cloned);
    memcpy(*cloned, trie, sizeof(Trie));
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TrieClone(trie->son[i], &(*cloned)->son[i]);
        }
    }
    return 0;
}

int TrieRelease(Trie *trie) {
    DLDevice cpu = {kDLCPU, 0};
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TrieRelease(trie->son[i]);
        }
    }
    return TVMDeviceFreeDataSpace(cpu, trie);
}
