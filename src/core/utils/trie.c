/*!
 * \file utils/trie.c
 * \brief the trie util implementation.
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#include <utils/trie.h>

/*! \brief this is a table for char to index (for all uint8_t )  for Trie */
const unsigned char char2index[] = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0,   255, 1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  255, 255, 255, 255, 255, 255, 255, 11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  255, 255, 255, 255,
    37,  255, 38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,
    55,  56,  57,  58,  59,  60,  61,  62,  63,  255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255};

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
    if (!dst->has_value) {
        dst->data = src->data;
        dst->has_value = true;
    }
    return 0;
}

void TVM_RT_WASM_TrieVisit(Trie *trie, void (*visit)(void **, void *), void *source_handle) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            if (trie->son[i]->has_value) {
                visit(&trie->son[i]->data, source_handle);
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
