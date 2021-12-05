/*!
 * \file runtime/utils/trie.h
 * \brief the trie util
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_TRIE_H
#define TVM_RT_TRIE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/utils/common.h>

typedef struct Trie Trie;

#define TRIE_INVALID_CHARSET (-1)
#define TRIE_NOT_FOUND (-2)
#define TRIE_SUCCESS 0

/*! \brief charset = 0-9,a-z,A-Z, _, : */
#define CHAR_SET_SIZE 64

/*! \brief this is a table for char to index (for all uint8_t ) */
extern const unsigned char char2index[];

extern const char index2char[];

/*! \brief the definition of Trie */
struct Trie {
    Trie *son[CHAR_SET_SIZE];
    void *data;
};

/**------------------------------------------public functions---------------------------------------------------------*/
/*!
 * \brief alloc a new Trie and init it
 * @param trie the pointer to receive new Trie
 * @return 0 if successful
 */
INLINE int TrieCreate(Trie **trie) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    int status = TVMDeviceAllocDataSpace(cpu, sizeof(Trie), 0, no_type, (void **)trie);
    memset(*trie, 0, sizeof(Trie));
    return status;
}

/*!
 * \brief insert a new <str,data> to trie
 * @param trie the instance of Trie
 * @param name the key
 * @param data the value
 * @return 0 if successful
 */
INLINE int TrieInsert(Trie *trie, const uint8_t *name, void *data) {
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

/*!
 * \brief insert all data from src to dst
 * @param dst the instance of dst Trie
 * @param src the instance of src Trie
 * @return 0 if successful
 */
INLINE int TrieInsertAll(Trie *dst, Trie *src) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (src->son[i]) {
            if (dst->son[i] == NULL) {
                TVMDeviceAllocDataSpace(cpu, sizeof(Trie), 0, no_type, (void **)&dst->son[i]);
                memset(dst->son[i], 0, sizeof(Trie));
            }
            TrieInsertAll(dst->son[i], src->son[i]);
        }
    }
    if (dst->data == NULL) {
        dst->data = src->data;
    }
    return 0;
}

/*!
 * \brief query the value for given key
 * @param trie the instance of Trie
 * @param name the key
 * @param data the pointer to receive value
 * @return TRIE_SUCCESS if successful and found, or TRIE_NOT_FOUND, TRIE_INVALID_CHARSET
 */
INLINE int TrieQuery(Trie *trie, const uint8_t *name, void **data) {
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

/*!
 * \brief Traversal the trie, and use visit function to visit every node's data
 * @param trie the instance of Trie
 * @param visit the visit function
 * @param source_handle for visit function
 */
INLINE void TrieVisit(Trie *trie, void (*visit)(char c, void *data, void *source_handle), void *source_handle) {
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            if (trie->son[i]->data) {
                visit(index2char[i], trie->son[i]->data, source_handle);
            }
            TrieVisit(trie->son[i], visit, source_handle);
        }
    }
}

/*!
 * \brief clone this Trie, and create a new Instance
 * @param trie the instance of Trie
 * @param cloned the pointer to receive new instance
 * @return 0 if successful
 */
INLINE int TrieClone(const Trie *trie, Trie **cloned) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(Trie), 0, no_type, (void **)cloned);
    memcpy(*cloned, trie, sizeof(Trie));
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TrieClone(trie->son[i], &(*cloned)->son[i]);
        }
    }
    return 0;
}

/*!
 * \brief free the instance of trie
 * @param trie the point to instance of Trie
 * @return 0 if successful
 */
INLINE int TrieRelease(Trie *trie) {
    DLDevice cpu = {kDLCPU, 0};
    for (int i = 0; i < CHAR_SET_SIZE; ++i) {
        if (trie->son[i]) {
            TrieRelease(trie->son[i]);
        }
    }
    return TVMDeviceFreeDataSpace(cpu, trie);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_TRIE_H
