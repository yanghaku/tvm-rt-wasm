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
 * \file internal/utils/common.h
 * \brief some common auxiliary definitions and functions
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_COMMON_H
#define TVM_RT_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__) || defined(__clang__)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

extern char global_buf[];

#define GLOBAL_BUF_SIZE 1024

#define SET_ERROR_RETURN(err, fmt, ...)                                                                                \
    do {                                                                                                               \
        sprintf(global_buf, "function[%s] " fmt, __FUNCTION__, ##__VA_ARGS__);                                         \
        return (err);                                                                                                  \
    } while (0)

#define SET_ERROR(fmt, ...)                                                                                            \
    do {                                                                                                               \
        sprintf(global_buf, "function[%s] " fmt, __FUNCTION__, ##__VA_ARGS__);                                         \
    } while (0)

#define MAX(x, y) ((x) > (y) ? (x) : (y))

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_COMMON_H
