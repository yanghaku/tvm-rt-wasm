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
 * \file internal/utils/trie.h
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
 * \brief query the value for given key
 * @param trie the instance of Trie
 * @param name the key
 * @param data the pointer to receive value
 * @return 0 if successful and found
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
