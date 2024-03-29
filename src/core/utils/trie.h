/**
 * @file utils/trie.h
 * @brief The trie data structure to save map<string, data>. The data can only be a pointer size.
 */

#ifndef TVM_RT_WASM_CORE_UTILS_TRIE_H_INCLUDE_
#define TVM_RT_WASM_CORE_UTILS_TRIE_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <string.h>

#include <device/cpu_memory.h>
#include <utils/common.h>

typedef struct Trie Trie;

#define TRIE_INVALID_CHARSET (-1)
#define TRIE_NOT_FOUND (-2)
#define TRIE_SUCCESS 0

/** @brief charset = 0-9,a-z,A-Z, _, : */
#define CHAR_SET_SIZE 64

/** @brief This is a table for char to index (for all uint8_t ) */
extern const unsigned char char2index[];

/** @brief Trie node */
struct Trie {
    Trie *son[CHAR_SET_SIZE];
    void *data;
    bool has_value;
};

/*-----------------------------------public functions---------------------------------------------*/
/**
 * @brief Create a new Trie and init it.
 * @param trie The pointer to save new Trie instance.
 */
INLINE void TVM_RT_WASM_TrieCreate(Trie **trie) {
    *trie = (Trie *)TVM_RT_WASM_HeapMemoryAlloc(sizeof(Trie));
    memset(*trie, 0, sizeof(Trie));
}

/**
 * @brief Insert a new <str, data> to this trie.
 * @param trie the instance of Trie
 * @param name the key
 * @param data the value
 * @return 0 if successful
 */
INLINE int TVM_RT_WASM_TrieInsert(Trie *trie, const uint8_t *name, void *data) {
    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            TVM_RT_SET_ERROR_RETURN(TRIE_INVALID_CHARSET, "Charset is invalid: %c(ascii=%d)", *name,
                                    *name);
        }
        if (trie->son[id] == NULL) {
            TVM_RT_WASM_TrieCreate(&trie->son[id]);
        }
        trie = trie->son[id];
        ++name;
    }
    trie->data = data;
    trie->has_value = true;
    return TRIE_SUCCESS;
}

/**
 * @brief Insert a new <str, data> to this trie.
 * @param trie the instance of Trie
 * @param name the key
 * @param len the len of key
 * @param data the value
 * @return 0 if successful
 */
INLINE int TVM_RT_WASM_TrieInsertWithLen(Trie *trie, const uint8_t *name, size_t len, void *data) {
    while (len--) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            TVM_RT_SET_ERROR_RETURN(TRIE_INVALID_CHARSET, "Charset is invalid: %c(ascii=%d)", *name,
                                    *name);
        }
        if (trie->son[id] == NULL) {
            TVM_RT_WASM_TrieCreate(&trie->son[id]);
        }
        trie = trie->son[id];
        ++name;
    }
    trie->data = data;
    trie->has_value = true;
    return TRIE_SUCCESS;
}

/**
 * @brief Query the value for given key.
 * @param trie the instance of Trie
 * @param name the key
 * @param data the pointer to receive value
 * @return TRIE_SUCCESS if successful and found, or TRIE_NOT_FOUND, TRIE_INVALID_CHARSET
 */
INLINE int TVM_RT_WASM_TrieQuery(Trie *trie, const uint8_t *name, void **data) {
    while (*name) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            return TRIE_INVALID_CHARSET;
        }
        if (trie->son[id]) {
            trie = trie->son[id];
            ++name;
        } else {
            *data = NULL;
            return TRIE_NOT_FOUND;
        }
    }
    if (likely(trie->has_value)) {
        *data = trie->data;
        return TRIE_SUCCESS;
    } else {
        return TRIE_NOT_FOUND;
    }
}

/**
 * @brief Query the value for given key
 * @param trie the instance of Trie
 * @param name the key
 * @param len the len of key
 * @param data the pointer to receive value
 * @return TRIE_SUCCESS if successful and found, or TRIE_NOT_FOUND, TRIE_INVALID_CHARSET
 */
INLINE int TVM_RT_WASM_TrieQueryWithLen(Trie *trie, const uint8_t *name, size_t len, void **data) {
    while (len--) {
        uint32_t id = char2index[*name];
        if (unlikely(id == 255)) {
            return TRIE_INVALID_CHARSET;
        }
        if (trie->son[id]) {
            trie = trie->son[id];
            ++name;
        } else {
            *data = NULL;
            return TRIE_NOT_FOUND;
        }
    }
    if (likely(trie->has_value)) {
        *data = trie->data;
        return TRIE_SUCCESS;
    } else {
        return TRIE_NOT_FOUND;
    }
}

/**
 * @brief Traversal the trie, and use visit function to visit every node's data.
 * @param trie the instance of Trie
 * @param visit the visit function
 * @param source_handle for visit function
 */
void TVM_RT_WASM_TrieVisit(Trie *trie, void (*visit)(void **p_data, void *source_handle),
                           void *source_handle);

/**
 * @brief Clone this Trie, and create a new Instance.
 * @param trie the instance of Trie
 * @param cloned the pointer to receive new instance
 * @return 0 if successful
 */
int TVM_RT_WASM_TrieClone(const Trie *trie, Trie **cloned);

/**
 * @brief Free the instance of trie.
 * @param trie the point to instance of Trie
 */
void TVM_RT_WASM_TrieRelease(Trie *trie);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_UTILS_TRIE_H_INCLUDE_
