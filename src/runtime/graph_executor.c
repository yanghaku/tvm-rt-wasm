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
 * \file src/runtime/graph_executor.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>
#include <tvm/internal/graph/graph_executor.h>
#include <tvm/internal/memory/memory.h>
#include <tvm/internal/utils/common.h>
#include <tvm/internal/utils/json.h>
#include <tvm/internal/utils/tensor.h>

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
int TVMGraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                           uint32_t num_dev, GraphManagerInterface **g) {
    DLDevice cpu = {kDLCPU, 0};
    memory_alloc(sizeof(GraphManagerInterface), cpu, (void **)g);

    (*g)->GetNumOfNodes = GraphExecutorGetNumOfNodes;
    (*g)->GetNodeName = GraphExecutorGetNodeName;
    (*g)->GetInputIndex = GraphExecutorGetInputIndex;
    (*g)->GetOutputIndex = GraphExecutorGetOutputIndex;
    (*g)->GetNumInputs = GraphExecutorGetNumInputs;
    (*g)->GetNumOutputs = GraphExecutorGetNumOutputs;
    (*g)->SetInput = GraphExecutorSetInput;
    (*g)->GetOutput = GraphExecutorGetOutput;
    (*g)->LoadParams = GraphExecutorLoadParams;
    (*g)->Run = GraphExecutorRun;
    (*g)->Release = GraphExecutorRelease;
    (*g)->Clone = GraphExecutorClone;

    memory_alloc(sizeof(GraphExecutor), cpu, &(*g)->graphHandle);
    memset((*g)->graphHandle, 0, sizeof(GraphExecutor));
    return GraphExecutorLoad(graph_json, module_handle, devices, num_dev, (*g)->graphHandle);
}

/*! \brief function for GraphExecutor_Load */
#define GRAPH_JSON_KEY_SIZE 32
int GraphExecutor_SetupStorage(GraphExecutor *);
int GraphExecutor_SetupOpExecs(GraphExecutor *);
int JsonReader_ReadGraphNodesArray(JsonReader *, GraphExecutor *);
int JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *, GraphExecutor *);
int JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *, GraphExecutor *);
int JsonReader_ReadGraphAttrObject(JsonReader *, GraphExecutor *);
int JsonReader_ReadGraphNodeRowPtrArray(JsonReader *, GraphExecutor *);

/*!
 * \brief init a new GraphExecutor from graph.json
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param graph the instance instance.
 * \return 0 if successful.
 */
int GraphExecutorLoad(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices, uint32_t num_dev,
                      GraphExecutor *graph) {

    if (unlikely(graph_json == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: graph json cannot be NULL");
    }
    if (unlikely(graph == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: graph executor cannot be NULL");
    }

    DLDevice cpu = {kDLCPU, 0};

    // Init JsonReader
    JsonReader *reader;
    int status = JsonReader_Create(graph_json, &reader);
    if (unlikely(status)) {
        SET_ERROR_RETURN(-1, "JsonReader Create fail, error code = %d\n", status);
    }

    char key[GRAPH_JSON_KEY_SIZE];
    // start to load graph
    while (JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
        if (!strcmp(key, "nodes")) {
            status = JsonReader_ReadGraphNodesArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
        } else if (!strcmp(key, "arg_nodes")) {
            status = JsonReader_ReadGraphInputNodeIndicesArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
        } else if (!strcmp(key, "heads")) {
            status = JsonReader_ReadGraphOutputNodeEntryArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
        } else if (!strcmp(key, "attrs")) {
            status = JsonReader_ReadGraphAttrObject(reader, graph);
            if (unlikely(status)) {
                return status;
            }
        } else if (!strcmp(key, "node_row_ptr")) {
            status = JsonReader_ReadGraphNodeRowPtrArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
        } else if (!strcmp(key, "metadata")) {
            break;
        } else {
            SET_ERROR_RETURN(-1, "unsupported Json key: %s", key);
        }
    }

    // release JsonReader
    status = JsonReader_Release(reader);

    // other member init
    status = memory_alloc(sizeof(DLDevice) * num_dev, cpu, (void **)&graph->devices);

    memcpy(graph->devices, devices, sizeof(DLDevice) * num_dev);
    graph->num_device = num_dev;
    graph->module_handle = module_handle;

    // init storage
    status = GraphExecutor_SetupStorage(graph);
    if (unlikely(status)) {
        return status;
    }
    // init operators
    return GraphExecutor_SetupOpExecs(graph);
}

/*!
 * \brief Get total number of nodes.
 * \param g The instance of GraphManagerInterface.
 * \return Total number of nodes.
 */
int GraphExecutorGetNumOfNodes(GraphManagerInterface *g) {
    CHECK_GraphManagerInterface(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_nodes;
}

/*!
 * \brief Get the name of node for given index.
 * \param g The instance of GraphManagerInterface.
 * \param nid the node index
 * \param name the pointer to receive string pointer
 * \return 0 if successful
 */
int GraphExecutorGetNodeName(GraphManagerInterface *g, uint32_t nid, const char **name) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(nid < 0 || nid > graph->num_nodes)) {
        SET_ERROR_RETURN(-1, "invalid argument: nid, expect it in range [0,%d), but given %d", graph->num_nodes, nid);
    }
    if (unlikely(name == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: name, must not be NULL");
    }
    *name = graph->nodes[nid].op_type;
    return 0;
}

/*!
 * \brief Get the input index given the name of input.
 * \param g The instance of GraphManagerInterface.
 * \param name The name of the input.
 * \return The index of input.
 */
int GraphExecutorGetInputIndex(GraphManagerInterface *g, const char *name) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(name == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: name, must not be NULL");
    }
    int index = -1;
    if (unlikely(TrieQuery(graph->inputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        SET_ERROR_RETURN(-1, "name(%s)is not FOUND", name);
    }
    return index;
}

/*!
 * \brief Get the output index given the name of output.
 * \param g The instance of GraphManagerInterface.
 * \param name The name of the output.
 * \return The index of output.
 */
int GraphExecutorGetOutputIndex(GraphManagerInterface *g, const char *name) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(name == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: name, must not be NULL");
    }
    int index = -1;
    if (unlikely(TrieQuery(graph->outputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        SET_ERROR_RETURN(-1, "name(%s)is not FOUND", name);
    }
    return index;
}

/*!
 * \brief get number of input tensors allocated.
 * \param g The instance of GraphManagerInterface.
 * \return integer number of tensors available to use.
 */
int GraphExecutorGetNumInputs(GraphManagerInterface *g) {
    CHECK_GraphManagerInterface(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_inputs_nodes;
}

/*!
 * \brief get number of output tensors allocated.
 * \param g The instance of GraphManagerInterface.
 * \return integer number of output tensors allocated.
 */
int GraphExecutorGetNumOutputs(GraphManagerInterface *g) {
    CHECK_GraphManagerInterface(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_outputs;
}

/*!
 * \brief set input to the graph based on name.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \param index the index of inputs.
 * \param data_in The input data.
 * \return 0 if successful
 */
int GraphExecutorSetInput(GraphManagerInterface *g, uint32_t index, const DLTensor *data_in) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    if (unlikely(index > graph->num_inputs_nodes)) {
        SET_ERROR_RETURN(-1, "invalid argument: index, expect it in range [0,%d), but given %d",
                         graph->num_inputs_nodes, index);
    }

    uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[index], 0);
    return DLTensor_CopyFromTo(data_in, graph->data_entry + eid, NULL);
}

/*!
 * \brief Return NDArray for given output index.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \param index The output index.
 * \param out The DLTensor corresponding to given output node index.
 * \return The result of this function execution.
 */
int GraphExecutorGetOutput(GraphManagerInterface *g, uint32_t index, DLTensor *data_out) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    if (unlikely(index > graph->num_outputs)) {
        SET_ERROR_RETURN(-1, "invalid argument: out_puts, expect it in range [0,%d), but given %d", graph->num_outputs,
                         index);
    }

    uint32_t eid = DATA_ENTRY_ID(graph, graph->outputs_nodes[index].node_id, graph->outputs_nodes[index].index);
    return DLTensor_CopyFromTo(graph->data_entry + eid, data_out, NULL);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
int GraphExecutorLoadParams(GraphManagerInterface *g, const char *param_blob, uint32_t param_size) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphManagerInterface.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int GraphExecutorRun(GraphManagerInterface *g) {
    CHECK_GraphManagerInterface(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    for (int i = 0; i < graph->num_nodes; ++i) {
        if (graph->nodeOps[i].exec) { // call backend function
            graph->nodeOps[i].exec(graph->nodeOps[i].arg_values, graph->nodeOps[i].arg_type_codes,
                                   graph->nodeOps[i].num_args, &graph->nodeOps[i].return_value,
                                   &graph->nodeOps[i].return_type_code, graph->module_handle);
        }
    }
    return 0;
}

/*!
 * \brief Release memory associated with the GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int GraphExecutorRelease(GraphManagerInterface **g) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Clone a new instance of GraphManagerInterface.
 * \param g The instance of GraphManagerInterface.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int GraphExecutorClone(GraphManagerInterface *g, GraphManagerInterface **cloned) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/**-----------------------------------------private functions---------------------------------------------------------*/

/*!
 * \brief setup storage for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int GraphExecutor_SetupStorage(GraphExecutor *graph) { return 0; }

/*!
 * \brief setup operators for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int GraphExecutor_SetupOpExecs(GraphExecutor *graph) { return 0; }

/*!
 * \brief load graph nodes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int JsonReader_ReadGraphNodesArray(JsonReader *reader, GraphExecutor *graph) {
    size_t node_size = 0;
    int status = JsonReader_ArrayLength(reader, &node_size);
    if (unlikely(status)) {
        SET_ERROR_RETURN(-1, "JsonReader get Node Array length Error");
    }
    graph->num_nodes = (uint32_t)node_size;

    DLDevice cpu = {kDLCPU, 0};
    memory_alloc(sizeof(GraphExecutorNode) * node_size, cpu, (void **)&graph->nodes);
    memset(graph->nodes, 0, sizeof(GraphExecutorNode) * node_size);

    uint32_t nid = 0;
    while (JsonReader_NextArrayItem(reader)) {
        if (unlikely(nid == node_size)) {
            SET_ERROR_RETURN(-1, "JsonReader Error: array length expect %d, but now >%d", nid, nid);
        }

        GraphExecutorNode *node = graph->nodes + nid;
        char key[GRAPH_JSON_KEY_SIZE];
        while (JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE)) {
            if (!strcmp(key, "op")) {
                status = JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(status)) {
                    SET_ERROR_RETURN(-1, "JsonReader Error: Parse string for GraphExecutorNode.op fail");
                }

                size_t str_len = strlen(global_buf);
                memory_alloc(str_len + 1, cpu, (void **)&node->op_type);
                strcpy((char *)node->op_type, global_buf);

            } else if (!strcmp(key, "name")) {
                status = JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(status)) {
                    SET_ERROR_RETURN(-1, "JsonReader Error: parse GraphExecutorNode.op fail");
                }

                size_t str_len = strlen(global_buf);
                memory_alloc(str_len + 1, cpu, (void **)&node->name);
                strcpy((char *)node->name, global_buf);
            } else if (!strcmp(key, "inputs")) {
                size_t inputs_num;
                status = JsonReader_ArrayLength(reader, &inputs_num);
                if (unlikely(status)) {
                    SET_ERROR_RETURN(-1, "JsonReader Error: get GraphExecutorNode.inputs length fail");
                }

                node->num_inputs = inputs_num;
                memory_alloc(sizeof(GraphExecutorNodeEntry) * inputs_num, cpu, (void **)&node->inputs);
                memset(node->inputs, 0, sizeof(GraphExecutorNodeEntry));
                int inputs_count = 0;
                while (JsonReader_NextArrayItem(reader)) {
                    if (unlikely(inputs_count == inputs_num)) {
                        SET_ERROR_RETURN(-1, "JsonReader Error: inputs array length expect %d, but now >%d",
                                         inputs_count, inputs_count);
                    }

                    size_t entry_size;
                    status = JsonReader_ArrayLength(reader, &entry_size);
                    if (unlikely(status)) {
                        SET_ERROR_RETURN(-1, "JsonReader Error: parse NodeEntry instance fail");
                    }
                    if (likely(entry_size == 2 || entry_size == 3)) {
                        JsonReader_NextArrayItem(reader);
                        status = JsonReader_Read_uint32(reader, &node->inputs[inputs_count].node_id);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Read uint32 fail for NodeEntry.node_id");
                        }
                        JsonReader_NextArrayItem(reader);
                        status = JsonReader_Read_uint32(reader, &node->inputs[inputs_count].index);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Read uint32 fail for NodeEntry.index");
                        }

                        if (likely(entry_size == 3)) { // version
                            JsonReader_NextArrayItem(reader);
                            uint32_t version_tmp;
                            status = JsonReader_Read_uint32(reader, &version_tmp);
                            if (unlikely(status)) {
                                SET_ERROR_RETURN(-1, "JsonReader Error: Read uint32 fail for NodeEntry.version");
                            }
                        }
                    } else {
                        SET_ERROR_RETURN(-1, "JsonReader Error: NodeEntry need len = 2/3, but given %zu", entry_size);
                    }
                    ++inputs_count;
                }

            } else if (!strcmp(key, "attr") || !strcmp(key, "attrs")) {
                while (JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE)) {
                    if (!strcmp(key, "func_name")) {
                        status = JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Parse string for Node Attrs.func_name fail");
                        }

                        size_t str_len = strlen(global_buf);
                        memory_alloc(str_len + 1, cpu, (void **)&node->func_name);
                        strcpy((char *)node->func_name, global_buf);

                    } else if (!strcmp(key, "num_inputs")) {
                        uint32_t num_inputs_tmp;
                        status = JsonReader_Read_uint32(reader, &num_inputs_tmp);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Parse uint32 for Node Attrs.num_inputs fail");
                        }
                        if (unlikely(node->inputs != NULL && num_inputs_tmp != node->num_inputs)) {
                            SET_ERROR_RETURN(-1,
                                             "JsonReader Data error: Node Attrs.num_inputs(%d) != Attrs.inputs.len(%d)",
                                             num_inputs_tmp, node->num_inputs);
                        }

                    } else if (!strcmp(key, "num_outputs")) {
                        status = JsonReader_Read_uint32(reader, &node->num_outputs);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Parse uint32 for Node Attrs.num_outputs fail");
                        }
                    } else if (!strcmp(key, "flatten_data")) {
                        status = JsonReader_Read_uint32(reader, &node->flatten_data);
                        if (unlikely(status)) {
                            SET_ERROR_RETURN(-1, "JsonReader Error: Parse uint32 for Node Attrs.flatten_data fail");
                        }
                    }
                }
            } else if (!strcmp(key, "control_deps")) {
                SET_ERROR_RETURN(-1, "unimplemented key: %s", key);
            } else {
                SET_ERROR_RETURN(-1, "unsupported key: %s", key);
            }
        }

        ++nid;
    }

    return 0;
}

/*!
 * \brief load graph input node indices from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *reader, GraphExecutor *graph) { return 0; }

/*!
 * \brief load graph output nodeEntry from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *reader, GraphExecutor *graph) { return 0; }

/*!
 * \brief load graph attributes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int JsonReader_ReadGraphAttrObject(JsonReader *reader, GraphExecutor *graph) { return 0; }

/*!
 * \brief load graph node_row_ptr from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
int JsonReader_ReadGraphNodeRowPtrArray(JsonReader *reader, GraphExecutor *graph) { return 0; }
