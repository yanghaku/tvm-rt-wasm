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
 * \file graph_manager_interface.h
 * \brief A Interface for graph manager
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */
#ifndef TVM_RT_GRAPH_MANAGER_INTERFACE_H
#define TVM_RT_GRAPH_MANAGER_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

typedef void *TVMGraphHandle;

typedef struct GraphManagerInterface GraphManagerInterface;

struct GraphManagerInterface {

    // handle the graph instance
    TVMGraphHandle graphHandle;

    // public functions
    /*!
     * \brief Get total number of nodes.
     * \param g The instance of GraphManagerInterface.
     * \return Total number of nodes.
     */
    int (*GetNumOfNodes)(GraphManagerInterface *g);

    /*!
     * \brief Get the name of node for given index.
     * \param g The instance of GraphManagerInterface.
     * \param nid the node index
     * \param name the pointer to receive string pointer
     * \return 0 if success
     */
    int (*GetNodeName)(GraphManagerInterface *g, uint32_t nid, const char **name);

    /*!
     * \brief Get the input index given the name of input.
     * \param g The instance of GraphManagerInterface.
     * \param name The name of the input.
     * \return The index of input.
     */
    int (*GetInputIndex)(GraphManagerInterface *g, const char *name);

    /*!
     * \brief Get the output index given the name of output.
     * \param g The instance of GraphManagerInterface.
     * \param name The name of the output.
     * \return The index of output.
     */
    int (*GetOutputIndex)(GraphManagerInterface *g, const char *name);

    /*!
     * \brief get number of input tensors allocated.
     * \param g The instance of GraphManagerInterface.
     * \return integer number of tensors available to use.
     */
    int (*GetNumInputs)(GraphManagerInterface *g);

    /*!
     * \brief get number of output tensors allocated.
     * \param g The instance of GraphManagerInterface.
     * \return integer number of output tensors allocated.
     */
    int (*GetNumOutputs)(GraphManagerInterface *g);

    /*!
     * \brief set input to the graph based on name.
     * \param g The instance of GraphManagerInterface.
     * \param executor The graph executor.
     * \param index the index of inputs.
     * \param data_in The input data.
     * \return 0 if successful
     */
    int (*SetInput)(GraphManagerInterface *g, uint32_t index, const DLTensor *data_in);

    /*!
     * \brief Return NDArray for given output index.
     * \param g The instance of GraphManagerInterface.
     * \param executor The graph executor.
     * \param index The output index.
     * \param out The DLTensor corresponding to given output node index.
     * \return The result of this function execution.
     */
    int (*GetOutput)(GraphManagerInterface *g, uint32_t index, DLTensor *data_out);

    /*!
     * \brief Load parameters from parameter blob.
     * \param g The instance of GraphManagerInterface.
     * \param executor The graph executor.
     * \param param_blob A binary blob of parameter.
     * \param param_size The parameter size.
     * \return The result of this function execution.
     */
    int (*LoadParams)(GraphManagerInterface *g, const char *param_blob, uint32_t param_size);

    /*!
     * \brief Execute the graph.
     * \param g The instance of GraphManagerInterface.
     * \param executor The graph executor.
     * \return 0 if success
     */
    int (*Run)(GraphManagerInterface *g);

    /*!
     * \brief Release memory associated with the GraphManagerInterface.
     * \param g The instance of GraphManagerInterface.
     * \param executor Pointer to graph executor.
     * \return 0 if successful
     */
    int (*Release)(GraphManagerInterface **g);

    /*!
     * \brief Clone a new instance of GraphManagerInterface.
     * \param g The instance of GraphManagerInterface.
     * \param cloned Pointer which receive the new instance.
     * \return 0 if successful
     */
    int (*Clone)(GraphManagerInterface *g, GraphManagerInterface **cloned);
};

/*!
 * \brief Allocate a new GraphManagerInterface and initialize it with GraphExecutor.
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
TVM_DLL int TVMGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                   uint32_t num_dev, GraphManagerInterface **g);

/*!
 * \brief Allocate a new GraphManagerInterface and initialize it with CUDAGraphExecutor.
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
TVM_DLL int TVMGraphExecutorCUDACreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                       uint32_t num_dev, GraphManagerInterface **g);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_GRAPH_MANAGER_INTERFACE_H
