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

typedef struct Trie Trie;

#define TRIE_INVALID_CHARSET (-1)
#define TRIE_NOT_FOUND (-2)
#define TRIE_SUCCESS 0

/*!
 * \brief alloc a new Trie and init it
 * @param trie the pointer to receive new Trie
 * @return 0 if successful
 */
int TrieCreate(Trie **trie);

/*!
 * \brief insert a new <str,data> to trie
 * @param trie the instance of Trie
 * @param name the key
 * @param data the value
 * @return 0 if successful
 */
int TrieInsert(Trie *trie, const uint8_t *name, void *data);

/*!
 * \brief insert all data from src to dst
 * @param dst the instance of dst Trie
 * @param src the instance of src Trie
 * @return 0 if successful
 */
int TrieInsertAll(Trie *dst, Trie *src);

/*!
 * \brief query the value for given key
 * @param trie the instance of Trie
 * @param name the key
 * @param data the pointer to receive value
 * @return TRIE_SUCCESS if successful and found, or TRIE_NOT_FOUND, TRIE_INVALID_CHARSET
 */
int TrieQuery(Trie *trie, const uint8_t *name, void **data);

/*!
 * \brief Traversal the trie, and use visit function to visit every node's data
 * @param trie the instance of Trie
 * @param visit the visit function
 * @param source_handle for visit function
 */
void TrieVisit(Trie *trie, void (*visit)(char c, void *data, void *source_handle), void *source_handle);

/*!
 * \brief clone this Trie, and create a new Instance
 * @param trie the instance of Trie
 * @param cloned the pointer to receive new instance
 * @return 0 if successful
 */
int TrieClone(const Trie *trie, Trie **cloned);

/*!
 * \brief free the instance of trie
 * @param trie the point to instance of Trie
 * @return 0 if successful
 */
int TrieRelease(Trie *trie);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_TRIE_H
