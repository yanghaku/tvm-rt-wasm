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
 * \file graph_interface_manager.h
 * \brief A Interface for graph executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_GRAPH_EXECUTOR_MANAGER_H
#define TVM_RT_GRAPH_EXECUTOR_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

typedef void *TVMGraphHandle;

typedef struct GraphExecutorManager GraphExecutorManager;

struct GraphExecutorManager {

    // handle the graph instance
    TVMGraphHandle graphHandle;

    // public functions
    /*!
     * \brief Get total number of nodes.
     * \param g The instance of GraphExecutorManager.
     * \return Total number of nodes.
     */
    int (*GetNumOfNodes)(GraphExecutorManager *g);

    /*!
     * \brief Get the name of node for given index.
     * \param g The instance of GraphExecutorManager.
     * \param nid the node index
     * \param name the pointer to receive string pointer
     * \return 0 if success
     */
    int (*GetNodeName)(GraphExecutorManager *g, uint32_t nid, const char **name);

    /*!
     * \brief Get the input index given the name of input.
     * \param g The instance of GraphExecutorManager.
     * \param name The name of the input.
     * \return The index of input.
     */
    int (*GetInputIndex)(GraphExecutorManager *g, const char *name);

    /*!
     * \brief Get the output index given the name of output.
     * \param g The instance of GraphExecutorManager.
     * \param name The name of the output.
     * \return The index of output.
     */
    int (*GetOutputIndex)(GraphExecutorManager *g, const char *name);

    /*!
     * \brief get number of input tensors allocated.
     * \param g The instance of GraphExecutorManager.
     * \return integer number of tensors available to use.
     */
    int (*GetNumInputs)(GraphExecutorManager *g);

    /*!
     * \brief get number of output tensors allocated.
     * \param g The instance of GraphExecutorManager.
     * \return integer number of output tensors allocated.
     */
    int (*GetNumOutputs)(GraphExecutorManager *g);

    /*!
     * \brief set input to the graph based on index.
     * \param g The instance of GraphExecutorManager.
     * \param executor The graph executor.
     * \param index the index of inputs.
     * \param data_in The input data.
     * \return 0 if successful
     */
    int (*SetInput)(GraphExecutorManager *g, uint32_t index, const DLTensor *data_in);

    /*!
     * \brief set input to the graph based on name.
     * \param g The instance of GraphExecutorManager.
     * \param executor The graph executor.
     * \param name the name string for node
     * \param data_in The input data.
     * \return 0 if successful
     */
    int (*SetInputByName)(GraphExecutorManager *g, const char *name, const DLTensor *data_in);

    /*!
     * \brief Return NDArray for given output index.
     * \param g The instance of GraphExecutorManager.
     * \param executor The graph executor.
     * \param index The output index.
     * \param out The DLTensor corresponding to given output node index.
     * \return The result of this function execution.
     */
    int (*GetOutput)(GraphExecutorManager *g, uint32_t index, DLTensor *data_out);

    /*!
     * \brief Load parameters from parameter blob.
     * \param g The instance of GraphExecutorManager.
     * \param executor The graph executor.
     * \param param_blob A binary blob of parameter.
     * \param param_size The parameter size.
     * \return The result of this function execution.
     */
    int (*LoadParams)(GraphExecutorManager *g, const char *param_blob, uint32_t param_size);

    /*!
     * \brief Execute the graph.
     * \param g The instance of GraphExecutorManager.
     * \param executor The graph executor.
     * \return 0 if success
     */
    int (*Run)(GraphExecutorManager *g);

    /*!
     * \brief Release memory associated with the GraphExecutorManager.
     * \param g The instance of GraphExecutorManager.
     * \param executor Pointer to graph executor.
     * \return 0 if successful
     */
    int (*Release)(GraphExecutorManager **g);

    /*!
     * \brief Clone a new instance of GraphExecutorManager.
     * \param g The instance of GraphExecutorManager.
     * \param cloned Pointer which receive the new instance.
     * \return 0 if successful
     */
    int (*Clone)(GraphExecutorManager *g, GraphExecutorManager **cloned);
};

/*!
 * \brief Create a GraphExecutorManager Instance for given type name
 * @param graph_name graph type name
 * @param graph_json the json for graph
 * @param module_handle TVM Module that exposes the functions to call.
 * @param devices runtime execution device.
 * @param num_dev the number of devices
 * @param g Pointer which receives a pointer to the newly-created instance.
 * @return 0 if successful
 */
TVM_DLL int GraphExecutorManagerFactory(const char *graph_name, const char *graph_json, TVMModuleHandle module_handle,
                                        const DLDevice *devices, uint32_t num_dev, GraphExecutorManager **g);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_GRAPH_EXECUTOR_MANAGER_H