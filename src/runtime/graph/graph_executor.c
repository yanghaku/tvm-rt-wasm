/*!
 * \file src/runtime/graph/graph_executor.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>
#include <tvm/runtime/device/cpu_memory.h>
#include <tvm/runtime/graph/graph_executor.h>
#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/json.h>
#include <tvm/runtime/utils/tensor_helper.h>

/*!
 * \brief Allocate a new GraphExecutorManager and initialize it with GraphExecutor.
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param g Pointer which receives a pointer to the newly-created instance.
 * \return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                    uint32_t num_dev, GraphExecutorManager **g) {
    *g = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorManager));

    (*g)->GetNumOfNodes = TVM_RT_WASM_GraphExecutorGetNumOfNodes;
    (*g)->GetNodeName = TVM_RT_WASM_GraphExecutorGetNodeName;
    (*g)->GetInputIndex = TVM_RT_WASM_GraphExecutorGetInputIndex;
    (*g)->GetOutputIndex = TVM_RT_WASM_GraphExecutorGetOutputIndex;
    (*g)->GetNumInputs = TVM_RT_WASM_GraphExecutorGetNumInputs;
    (*g)->GetNumOutputs = TVM_RT_WASM_GraphExecutorGetNumOutputs;
    (*g)->SetInput = TVM_RT_WASM_GraphExecutorSetInput;
    (*g)->SetInputByName = TVM_RT_WASM_GraphExecutorSetInputByName;
    (*g)->GetOutput = TVM_RT_WASM_GraphExecutorGetOutput;
    (*g)->LoadParams = TVM_RT_WASM_GraphExecutorLoadParams;
    (*g)->Run = TVM_RT_WASM_GraphExecutorRun;
    (*g)->Release = TVM_RT_WASM_GraphExecutorRelease;
    (*g)->Clone = TVM_RT_WASM_GraphExecutorClone;

    (*g)->graphHandle = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutor));
    memset((*g)->graphHandle, 0, sizeof(GraphExecutor));
    return TVM_RT_WASM_GraphExecutorLoad(graph_json, module_handle, devices, num_dev, (*g)->graphHandle);
}

/*! \brief function for GraphExecutor_Load */
#define GRAPH_JSON_KEY_SIZE 32
static int TVM_RT_WASM_GraphExecutor_SetupStorage(GraphExecutor *);
static int TVM_RT_WASM_GraphExecutor_SetupOpExecs(GraphExecutor *);
static int TVM_RT_WASM_JsonReader_ReadGraphNodesArray(JsonReader *, GraphExecutor *);
static int TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *, GraphExecutor *);
static int TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *, GraphExecutor *);
static int TVM_RT_WASM_JsonReader_ReadGraphAttrObject(JsonReader *, GraphExecutor *);
static int TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(JsonReader *, GraphExecutor *);

/*!
 * \brief init a new GraphExecutor from graph.json
 *
 * \param graph_json JSON-encoded graph.
 * \param module_handle TVM Module that exposes the functions to call.
 * \param devices runtime execution device.
 * \param num_dev the number of devices
 * \param graph the instance.
 * \return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorLoad(const char *graph_json, TVMModuleHandle module_handle, const DLDevice *devices,
                                  uint32_t num_dev, GraphExecutor *graph) {
    // Init JsonReader
    JsonReader *reader;
    TVM_RT_WASM_JsonReader_Create(graph_json, &reader);
    int status;

    char key[GRAPH_JSON_KEY_SIZE];
    // start to load graph
    int bitmask = 0;
    while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
        if (!strcmp(key, "nodes")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphNodesArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
            bitmask |= 1;
        } else if (!strcmp(key, "arg_nodes")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
            bitmask |= 2;
        } else if (!strcmp(key, "heads")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
            bitmask |= 4;
        } else if (!strcmp(key, "attrs")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphAttrObject(reader, graph);
            if (unlikely(status)) {
                return status;
            }
            bitmask |= 8;
        } else if (!strcmp(key, "node_row_ptr")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(reader, graph);
            if (unlikely(status)) {
                return status;
            }
            bitmask |= 16;
        } else if (!strcmp(key, "metadata")) {
            break;
        } else {
            SET_ERROR_RETURN(-1, "unsupported Json key: %s", key);
        }
    }
    if (unlikely(bitmask != (1 | 2 | 4 | 8 | 16))) {
        SET_ERROR_RETURN(-1, "GraphExecutor need key: nodes,arg_nodes,heads,attrs,node_row_ptr");
    }

    // release JsonReader
    TVM_RT_WASM_JsonReader_Release(reader);

    // other member init
    graph->devices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDevice) * num_dev);
    memcpy(graph->devices, devices, sizeof(DLDevice) * num_dev);
    graph->num_device = num_dev;
    graph->module_handle = module_handle;

    if (unlikely(graph->num_data_entry == 0)) {
        SET_ERROR_RETURN(-1, "the number of graph data_entry cannot be 0, at least 1");
    }

    TVM_RT_WASM_TrieCreate(&graph->inputs_map);
    for (uintptr_t i = 0; i < graph->num_inputs_nodes; ++i) {
        uint32_t nid = graph->inputs_nodes[i];
        status = TVM_RT_WASM_TrieInsert(graph->inputs_map, (const uint8_t *)graph->nodes[nid].name, (void *)i);
        if (status) {
            SET_ERROR_RETURN(-1, "inputs_map: insert data fail");
        }
    }

    TVM_RT_WASM_TrieCreate(&graph->outputs_map);
    for (uintptr_t i = 0; i < graph->num_outputs; ++i) {
        uint32_t nid = graph->outputs_nodes[i].node_id;
        status = TVM_RT_WASM_TrieInsert(graph->outputs_map, (const uint8_t *)graph->nodes[nid].name, (void *)i);
        if (status) {
            SET_ERROR_RETURN(-1, "outputs_map: insert data fail");
        }
    }

    // init storage
    status = TVM_RT_WASM_GraphExecutor_SetupStorage(graph);
    if (unlikely(status)) {
        return status;
    }
    // init operators
    return TVM_RT_WASM_GraphExecutor_SetupOpExecs(graph);
}

/*!
 * \brief Get total number of nodes.
 * \param g The instance of GraphExecutorManager.
 * \return Total number of nodes.
 */
int TVM_RT_WASM_GraphExecutorGetNumOfNodes(GraphExecutorManager *g) {
    CHECK_GraphExecutorManager(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_nodes;
}

/*!
 * \brief Get the name of node for given index.
 * \param g The instance of GraphExecutorManager.
 * \param nid the node index
 * \param name the pointer to receive string pointer
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorGetNodeName(GraphExecutorManager *g, uint32_t nid, const char **name) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(nid > graph->num_nodes)) {
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
 * \param g The instance of GraphExecutorManager.
 * \param name The name of the input.
 * \return The index of input.
 */
int TVM_RT_WASM_GraphExecutorGetInputIndex(GraphExecutorManager *g, const char *name) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(name == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: name, must not be NULL");
    }
    intptr_t index = -1;
    if (unlikely(TVM_RT_WASM_TrieQuery(graph->inputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        SET_ERROR_RETURN(-1, "name(%s)is not FOUND in input nodes", name);
    }
    return index;
}

/*!
 * \brief Get the output index given the name of output.
 * \param g The instance of GraphExecutorManager.
 * \param name The name of the output.
 * \return The index of output.
 */
int TVM_RT_WASM_GraphExecutorGetOutputIndex(GraphExecutorManager *g, const char *name) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;
    if (unlikely(name == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: name, must not be NULL");
    }
    intptr_t index = -1;
    if (unlikely(TVM_RT_WASM_TrieQuery(graph->outputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        SET_ERROR_RETURN(-1, "name(%s)is not FOUND in output nodes", name);
    }
    return index;
}

/*!
 * \brief get number of input tensors allocated.
 * \param g The instance of GraphExecutorManager.
 * \return integer number of tensors available to use.
 */
int TVM_RT_WASM_GraphExecutorGetNumInputs(GraphExecutorManager *g) {
    CHECK_GraphExecutorManager(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_inputs_nodes;
}

/*!
 * \brief get number of output tensors allocated.
 * \param g The instance of GraphExecutorManager.
 * \return integer number of output tensors allocated.
 */
int TVM_RT_WASM_GraphExecutorGetNumOutputs(GraphExecutorManager *g) {
    CHECK_GraphExecutorManager(g);
    return (int)((GraphExecutor *)g->graphHandle)->num_outputs;
}

/*!
 * \brief set input to the graph based on name.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param index the index of inputs.
 * \param data_in The input data.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorSetInput(GraphExecutorManager *g, uint32_t index, const DLTensor *data_in) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    if (unlikely(index > graph->num_inputs_nodes)) {
        SET_ERROR_RETURN(-1, "invalid argument: index, expect it in range [0,%d), but given %d",
                         graph->num_inputs_nodes, index);
    }
    uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[index], 0);
    return TVMDeviceCopyDataFromTo((DLTensor *)data_in, &graph->data_entry[eid].dl_tensor, NULL);
}

/*!
 * \brief set input to the graph based on name.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param name the name string for node
 * \param data_in The input data.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorSetInputByName(GraphExecutorManager *g, const char *name, const DLTensor *data_in) {
    int index = TVM_RT_WASM_GraphExecutorGetInputIndex(g, name);
    if (unlikely(index == -1)) {
        return index;
    }
    return TVM_RT_WASM_GraphExecutorSetInput(g, index, data_in);
}

/*!
 * \brief Return NDArray for given output index.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param index The output index.
 * \param out The DLTensor corresponding to given output node index.
 * \return The result of this function execution.
 */
int TVM_RT_WASM_GraphExecutorGetOutput(GraphExecutorManager *g, uint32_t index, DLTensor *data_out) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    if (unlikely(index > graph->num_outputs)) {
        SET_ERROR_RETURN(-1, "invalid argument: out_puts, expect it in range [0,%d), but given %d", graph->num_outputs,
                         index);
    }

    uint32_t eid = DATA_ENTRY_ID(graph, graph->outputs_nodes[index].node_id, graph->outputs_nodes[index].index);
    return TVMDeviceCopyDataFromTo(&graph->data_entry[eid].dl_tensor, data_out, NULL);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
int TVM_RT_WASM_GraphExecutorLoadParams(GraphExecutorManager *g, const char *param_blob, uint32_t param_size) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    if (unlikely(param_blob == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: param_blob cannot be null");
    }
    if (unlikely(param_size < sizeof(uint64_t) * 2)) {
        SET_ERROR_RETURN(-1, "invalid argument: param_size is too short, at least %zu", sizeof(uint64_t) * 2);
    }
    if (unlikely(*((uint64_t *)param_blob) != kTVMNDArrayListMagic)) {
        SET_ERROR_RETURN(-1, "invalid param blob: magic error, expected %llu, given %llu", kTVMNDArrayListMagic,
                         *((uint64_t *)param_blob));
    }
    const char *blob = param_blob + sizeof(uint64_t) + sizeof(uint64_t); // magic(8 bytes), reserved(8 bytes)

    uint64_t name_num;
    memcpy(&name_num, blob, sizeof(name_num));
    blob += sizeof(name_num);
    char *name = (char *)blob;

    // scan names
    for (uint32_t i = 0; i < (uint32_t)name_num; ++i) {
        uint32_t str_len = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t) + str_len;
        if (unlikely(str_len == 0)) {
            SET_ERROR_RETURN(-1, "invalid param blob: node name cannot be \"\"");
        }
    }

    uint64_t arr_num;
    memcpy(&arr_num, blob, sizeof(arr_num));
    blob += sizeof(arr_num);

    if (unlikely(name_num != arr_num)) {
        SET_ERROR_RETURN(-1, "invalid param blob: name_num(%llu) != arr_num(%llu)", name_num, arr_num);
    }

    // scan name and load param
    for (uint32_t i = 0; i < (uint32_t)arr_num; ++i) {
        uint32_t str_len = (uint32_t) * (uint64_t *)name;
        name += sizeof(uint64_t);

        intptr_t index = -1;
        if (unlikely(TVM_RT_WASM_TrieQueryWithLen(graph->inputs_map, (const uint8_t *)name, str_len, (void **)&index) ==
                     TRIE_NOT_FOUND)) {
            SET_ERROR_RETURN(-1, "invalid param blob: param node name(%s) not found", name);
        }

        uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[index], 0);
        if (unlikely(eid >= graph->num_data_entry)) {
            SET_ERROR_RETURN(-1, "Error, entry id (%u) is greater than the number of data entry (%u)", eid,
                             graph->num_data_entry);
        }

        int status = TVM_RT_WASM_DLTensor_LoadDataFromBinary(&graph->data_entry[eid].dl_tensor, &blob);
        if (unlikely(status)) {
            return status;
        }

        // point to next name
        name += str_len;
    }

    return 0;
}

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorRun(GraphExecutorManager *g) {
    CHECK_GraphExecutorManager(g);
    GraphExecutor *graph = (GraphExecutor *)g->graphHandle;

    for (uint32_t i = 0; i < graph->num_nodes; ++i) {
        PackedFunction *pf = graph->nodeOps[i].exec;
        if (pf) { // call function handle
            pf->exec(graph->nodeOps[i].arg_values, graph->nodeOps[i].arg_type_codes, graph->nodeOps[i].num_args,
                     &graph->nodeOps[i].return_value, &graph->nodeOps[i].return_type_code, pf);
        }
    }
    return 0;
}

/*!
 * \brief Release memory associated with the GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorRelease(GraphExecutorManager **g) {
    if (unlikely(g == NULL)) {
        SET_ERROR_RETURN(-1, "invalid param: the GraphExecutorManager pointer cannot be NULL");
    }
    CHECK_GraphExecutorManager(*g);
    GraphExecutor *graph = (GraphExecutor *)(*g)->graphHandle;

    // free nodes
    for (uint32_t nid = 0; nid < graph->num_nodes; ++nid) {
        if (likely(graph->nodes[nid].op_type)) {
            TVM_RT_WASM_HeapMemoryFree((void *)graph->nodes[nid].op_type);
        }
        if (likely(graph->nodes[nid].name)) {
            TVM_RT_WASM_HeapMemoryFree((void *)graph->nodes[nid].name);
        }
        if (likely(graph->nodes[nid].func_name)) {
            TVM_RT_WASM_HeapMemoryFree((void *)graph->nodes[nid].func_name);
        }
        if (likely(graph->nodes[nid].inputs)) {
            TVM_RT_WASM_HeapMemoryFree((void *)graph->nodes[nid].inputs);
        }
    }
    TVM_RT_WASM_HeapMemoryFree(graph->nodes);

    // free node operators
    TVM_RT_WASM_HeapMemoryFree(graph->nodeOps);
    TVM_RT_WASM_HeapMemoryFree(graph->node_op_arg_type_storage);
    TVM_RT_WASM_HeapMemoryFree(graph->node_op_arg_value_storage);

    // free inputs nodes
    if (graph->inputs_nodes) {
        TVM_RT_WASM_HeapMemoryFree(graph->inputs_nodes);
    }

    // free output nodes entry
    if (graph->outputs_nodes) {
        TVM_RT_WASM_HeapMemoryFree(graph->outputs_nodes);
    }

    // free data entry + storage + storage_is_linked_param
    for (uint32_t eid = 0; eid < graph->num_data_entry; ++eid) {
        uint32_t sid = graph->data_entry[eid].storage_id;
        if (!graph->storages[sid].is_linked_param) {
            TVMDeviceFreeDataSpace(graph->data_entry[eid].dl_tensor.device, graph->storages[sid].storage);
            graph->storages[sid].is_linked_param = 1;
        }
        TVM_RT_WASM_HeapMemoryFree(graph->data_entry[eid].dl_tensor.shape);
    }
    if (graph->num_data_entry) {
        TVM_RT_WASM_HeapMemoryFree(graph->data_entry);
        TVM_RT_WASM_HeapMemoryFree(graph->storages);
    }

    // free input map and output map
    if (graph->inputs_map) {
        TVM_RT_WASM_TrieRelease(graph->inputs_map);
    }
    if (graph->outputs_map) {
        TVM_RT_WASM_TrieRelease(graph->outputs_map);
    }

    // devices
    if (graph->devices) {
        TVM_RT_WASM_HeapMemoryFree(graph->devices);
    }

    // free itself
    TVM_RT_WASM_HeapMemoryFree(graph);
    TVM_RT_WASM_HeapMemoryFree(*g);

    return 0;
}

/*!
 * \brief Clone a new instance of GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorClone(GraphExecutorManager *g, GraphExecutorManager **cloned) {
    CHECK_GraphExecutorManager(g);
    if (unlikely(cloned == NULL)) {
        SET_ERROR_RETURN(-1, "invalid argument: cloned pointer cannot be NULL");
    }

    *cloned = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorManager));
    memcpy(*cloned, g, sizeof(GraphExecutorManager));
    (*cloned)->graphHandle = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutor));
    memcpy((*cloned)->graphHandle, g->graphHandle, sizeof(GraphExecutor));

    GraphExecutor *new_g = (GraphExecutor *)(*cloned)->graphHandle;
    GraphExecutor *old_g = (GraphExecutor *)g->graphHandle;

    // deep copy

    // nodes
    new_g->nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNode) * new_g->num_nodes);
    for (uint32_t nid = 0; nid < new_g->num_nodes; ++nid) {
        new_g->nodes[nid].flatten_data = old_g->nodes[nid].flatten_data;
        new_g->nodes[nid].num_inputs = old_g->nodes[nid].num_inputs;
        new_g->nodes[nid].num_outputs = old_g->nodes[nid].num_outputs;
        // op type
        new_g->nodes[nid].op_type = TVM_RT_WASM_HeapMemoryAlloc(sizeof(char) * strlen(old_g->nodes[nid].op_type) + 1);
        strcpy((char *)new_g->nodes[nid].op_type, old_g->nodes[nid].op_type);
        // name
        new_g->nodes[nid].name = TVM_RT_WASM_HeapMemoryAlloc(sizeof(char) * strlen(old_g->nodes[nid].name) + 1);
        strcpy((char *)new_g->nodes[nid].name, old_g->nodes[nid].name);
        // func_name
        new_g->nodes[nid].func_name =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(char) * strlen(old_g->nodes[nid].func_name) + 1);
        strcpy((char *)new_g->nodes[nid].func_name, old_g->nodes[nid].func_name);
        // inputs
        new_g->nodes[nid].inputs =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * new_g->nodes[nid].num_inputs);
        memcpy(new_g->nodes[nid].inputs, old_g->nodes[nid].inputs,
               sizeof(GraphExecutorNodeEntry) * new_g->nodes[nid].num_inputs);
    }

    // input nodes
    new_g->inputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * new_g->num_inputs_nodes);
    memcpy(new_g->inputs_nodes, old_g->inputs_nodes, sizeof(uint32_t) * new_g->num_inputs_nodes);

    // out nodes entry
    new_g->outputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * new_g->num_outputs);
    memcpy(new_g->outputs_nodes, old_g->outputs_nodes, sizeof(uint32_t) * new_g->num_outputs);

    // input and output map
    TVM_RT_WASM_TrieClone(old_g->inputs_map, &new_g->inputs_map);
    TVM_RT_WASM_TrieClone(old_g->outputs_map, &new_g->outputs_map);

    // data entry and is linked param
    new_g->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * new_g->num_data_entry);
    memcpy(new_g->data_entry, old_g->data_entry, sizeof(DataEntry) * new_g->num_data_entry);
    uint32_t num_storage = 0;
    for (uint32_t i = 0; i < new_g->num_data_entry; ++i) {
        num_storage = MAX(new_g->data_entry[i].storage_id, num_storage);
    }
    ++num_storage;
    new_g->storages = TVM_RT_WASM_HeapMemoryAlloc(sizeof(StorageEntry) * num_storage);
    memset(new_g->storages, 0, sizeof(StorageEntry) * num_storage);

    // setup storage !!!
    uint32_t *tmp_storage_size;
    tmp_storage_size = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(uint32_t) * num_storage);
    memset(tmp_storage_size, 0, sizeof(uint32_t) * num_storage);
    for (uint32_t eid = 0; eid < new_g->num_data_entry; ++eid) {
        uint32_t sid = new_g->data_entry[eid].storage_id;
        if (new_g->storages[sid].is_linked_param) {
            new_g->storages[sid] = old_g->storages[sid];
            continue;
        }
        uint32_t size = (uint32_t)TVM_RT_WASM_DLTensor_GetDataBytes(&new_g->data_entry[eid].dl_tensor);
        tmp_storage_size[sid] = MAX(tmp_storage_size[sid], size);
    }
    DLDataType no_type = {0, 0, 0};
    for (uint32_t eid = 0; eid < new_g->num_data_entry; ++eid) {
        uint32_t sid = new_g->data_entry[eid].storage_id;
        if (new_g->storages == NULL) {
            TVMDeviceAllocDataSpace(new_g->data_entry[eid].dl_tensor.device, tmp_storage_size[sid], 0, no_type,
                                    (void **)(new_g->storages + sid));
            TVMDeviceCopyDataFromTo(&old_g->data_entry[eid].dl_tensor, &new_g->data_entry[eid].dl_tensor, NULL);
        } else {
            new_g->data_entry[eid].dl_tensor.data = new_g->storages[sid].storage;
        }
    }
    TVM_RT_WASM_WorkplaceMemoryFree(tmp_storage_size);

    // node ops
    new_g->nodeOps = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeOp) * new_g->num_nodes);
    memcpy(new_g->nodeOps, old_g->nodeOps, sizeof(GraphExecutorNodeOp) * new_g->num_nodes);
    uint32_t num_op = 0;
    for (uint32_t nid = 0; nid < new_g->num_nodes; ++nid) {
        num_op += new_g->nodeOps->num_args;
    }
    new_g->node_op_arg_value_storage = TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMValue) * num_op);
    new_g->node_op_arg_type_storage = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int) * num_op);
    // setup operators !!!
    TVMValue *alloc_value = new_g->node_op_arg_value_storage;
    int *alloc_type = new_g->node_op_arg_type_storage;
    for (uint32_t nid = 0; nid < new_g->num_nodes; ++nid) {
        GraphExecutorNode *node = new_g->nodes + nid;
        GraphExecutorNodeOp *nodeOp = new_g->nodeOps + nid;
        nodeOp->arg_values = alloc_value;
        nodeOp->arg_type_codes = alloc_type;
        alloc_type += nodeOp->num_args;
        alloc_value += nodeOp->num_args;
        for (uint32_t i = 0; i < node->num_inputs; ++i) {
            int eid = DATA_ENTRY_ID(new_g, node->inputs[i].node_id, node->inputs[i].index);
            nodeOp->arg_values[i].v_handle = &new_g->data_entry[eid];
            nodeOp->arg_type_codes[i] = kTVMDLTensorHandle;
        }
        for (uint32_t i = 0; i < node->num_outputs; ++i) {
            int eid = DATA_ENTRY_ID(new_g, nid, i);
            nodeOp->arg_values[node->num_inputs + i].v_handle = &new_g->data_entry[eid];
            nodeOp->arg_type_codes[node->num_inputs + i] = kTVMDLTensorHandle;
        }
    }

    return 0;
}

/**-----------------------------------------private functions---------------------------------------------------------*/

/*!
 * \brief setup storage for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutor_SetupStorage(GraphExecutor *graph) {
    DLDataType no_type = {0, 0, 0};
    size_t *storage_size;
    DLDevice *storage_device;

    // get the number of storage:  max(sid) + 1
    uint32_t num_storage = 0;
    for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
        num_storage = MAX(num_storage, graph->data_entry[i].storage_id);
    }
    ++num_storage;
    graph->storages = TVM_RT_WASM_HeapMemoryAlloc(sizeof(StorageEntry) * num_storage);
    memset(graph->storages, 0, sizeof(StorageEntry) * num_storage);

    // get the data size for every storage
    storage_size = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(size_t) * num_storage);
    memset(storage_size, 0, sizeof(size_t) * num_storage);
    for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
        size_t now_size = TVM_RT_WASM_DLTensor_GetDataSize(graph->data_entry[i].dl_tensor.shape,
                                                           (int)graph->data_entry[i].dl_tensor.ndim);
        now_size =
            ((graph->data_entry[i].dl_tensor.dtype.bits * graph->data_entry[i].dl_tensor.dtype.lanes + 7U) / 8U) *
            now_size;
        if (unlikely(now_size == 0)) {
            SET_ERROR_RETURN(-1, "shape cannot contains 0 in the %d shape", i);
        }
        uint32_t sid = graph->data_entry[i].storage_id;
        storage_size[sid] = MAX(storage_size[sid], now_size);
    }

    // get the device for every storage
    storage_device = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(DLDevice) * num_storage);
    memset(storage_device, 0xFF, sizeof(DLDevice) * num_storage);
    for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
        uint32_t sid = graph->data_entry[i].storage_id;
        if ((int)storage_device[sid].device_type == -1) {
            storage_device[sid].device_type = graph->data_entry[i].dl_tensor.device.device_type;
        } else {
            if (unlikely(storage_device[sid].device_type != graph->data_entry[i].dl_tensor.device.device_type)) {
                SET_ERROR_RETURN(-1, "The same storage requires the same device_type, but given %d and %d",
                                 storage_device[sid].device_type, graph->data_entry[i].dl_tensor.device.device_type);
            }
        }
    }
    for (uint32_t i = 0; i < num_storage; ++i) {
        // find the fit device
        for (uint32_t x = 0; x < graph->num_device; ++x) {
            if (graph->devices[x].device_type == storage_device[i].device_type) {
                storage_device[i].device_id = graph->devices[x].device_id;
                break;
            }
        }
        if (unlikely(storage_device[i].device_id == -1)) {
            storage_device[i] = graph->devices[0];
        }
    }

    // find linked param
    static const char *lookup_linked_param_func_name = "_lookup_linked_param";
    TVMFunctionHandle func;
    int status = TVMModGetFunction(graph->module_handle, lookup_linked_param_func_name, 1, &func);
    if (status == 0) {
        TVMValue arg_val, ret_val;
        int arg_type, ret_type;
        arg_type = kTVMArgInt;
        for (uint32_t i = 0; i < num_storage; ++i) {
            arg_val.v_int64 = i;
            status = TVMFuncCall(func, &arg_val, &arg_type, 1, &ret_val, &ret_type);
            if (likely(status == 0 && ret_val.v_handle != NULL)) {
                graph->storages[i].is_linked_param = 1;
                graph->storages[i].storage = ret_val.v_handle;
            }
        }
    }

    // alloc memory for storage
    for (uint32_t i = 0; i < num_storage; ++i) {
        if (graph->storages[i].is_linked_param == 0) {
            TVMDeviceAllocDataSpace(storage_device[i], storage_size[i], 0, no_type, &(graph->storages[i].storage));
        }
    }

    // set up the data_entry
    for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
        graph->data_entry[i].dl_tensor.data = graph->storages[graph->data_entry[i].storage_id].storage;
    }

    TVM_RT_WASM_WorkplaceMemoryFree(storage_device);
    TVM_RT_WASM_WorkplaceMemoryFree(storage_size);
    return 0;
}

/*!
 * \brief setup operators for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutor_SetupOpExecs(GraphExecutor *graph) {
    // init memory
    graph->nodeOps = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeOp) * graph->num_nodes);
    memset(graph->nodeOps, 0, sizeof(GraphExecutorNodeOp) * graph->num_nodes);

    uint32_t num_op_storage = 0;
    for (uint32_t nid = 0; nid < graph->num_nodes; ++nid) {
        graph->nodeOps[nid].num_args = (int)(graph->nodes[nid].num_inputs + graph->nodes[nid].num_outputs);
        num_op_storage += graph->nodeOps[nid].num_args;
    }
    graph->node_op_arg_type_storage = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int) * num_op_storage);
    graph->node_op_arg_value_storage = TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMValue) * num_op_storage);
    TVMValue *alloc_value = graph->node_op_arg_value_storage;
    int *alloc_type = graph->node_op_arg_type_storage;

    for (uint32_t nid = 0; nid < graph->num_nodes; ++nid) {
        GraphExecutorNode *node = graph->nodes + nid;
        if (strcmp(node->op_type, "tvm_op") == 0) {
            GraphExecutorNodeOp *nodeOp = graph->nodeOps + nid;
            nodeOp->arg_values = alloc_value;
            nodeOp->arg_type_codes = alloc_type;
            alloc_value += nodeOp->num_args;
            alloc_type += nodeOp->num_args;

            for (uint32_t i = 0; i < node->num_inputs; ++i) {
                int eid = DATA_ENTRY_ID(graph, node->inputs[i].node_id, node->inputs[i].index);
                nodeOp->arg_values[i].v_handle = &graph->data_entry[eid].dl_tensor;
                nodeOp->arg_type_codes[i] = kTVMDLTensorHandle;
            }
            for (uint32_t i = 0; i < node->num_outputs; ++i) {
                int eid = DATA_ENTRY_ID(graph, nid, i);
                nodeOp->arg_values[node->num_inputs + i].v_handle = &graph->data_entry[eid].dl_tensor;
                nodeOp->arg_type_codes[node->num_inputs + i] = kTVMDLTensorHandle;
            }

            if (strcmp(node->func_name, "__nop") == 0) {
                nodeOp->exec = NULL;
                continue;
            }
            int status = TVMModGetFunction(graph->module_handle, node->func_name, 1, &nodeOp->exec);
            if (unlikely(status)) {
                SET_ERROR_RETURN(-1, "cannot find func from module, name = %s", node->func_name);
            }

        } else if (strcmp(node->op_type, "null") == 0) {
            continue;
        } else {
            SET_ERROR_RETURN(-1, "unsupported graph node op_type: %s", node->op_type);
        }
    }
    return 0;
}

#define JSON_READER_ERROR_PREFIX "Graph JsonReader Error:"
#define JSON_ERROR(fmt, ...) SET_ERROR_RETURN(-1, "%s" fmt, JSON_READER_ERROR_PREFIX, ##__VA_ARGS__)

/*! \brief json next array item exist check */
#define ARRAY_CHECK_NEXT_EXISTS(reader, fmt, ...)                                                                      \
    do {                                                                                                               \
        status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);                                                         \
        if (unlikely(status != 1)) {                                                                                   \
            JSON_ERROR(fmt, ##__VA_ARGS__);                                                                            \
        }                                                                                                              \
    } while (0)

/*! \brief json next array item no-exist check */
#define ARRAY_CHECK_NEXT_NON_EXISTS(reader, fmt, ...)                                                                  \
    do {                                                                                                               \
        status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);                                                         \
        if (unlikely(status != 0)) {                                                                                   \
            JSON_ERROR(fmt, ##__VA_ARGS__);                                                                            \
        }                                                                                                              \
    } while (0)

/*! \brief parse the digit string */
#define STR_DIGIT_TO_UINT(str, str_len, num)                                                                           \
    do {                                                                                                               \
        for (int i = 0; i < (str_len); ++i) {                                                                          \
            (num) = ((num) << 3) + ((num) << 1) + (str)[i] - '0';                                                      \
        }                                                                                                              \
    } while (0)

/*!
 * \brief load graph nodes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphNodesArray(JsonReader *reader, GraphExecutor *graph) {
    size_t node_size = 0;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &node_size);
    if (unlikely(status)) {
        JSON_ERROR("parse Node Array length fail");
    }
    graph->num_nodes = (uint32_t)node_size;
    if (unlikely(node_size == 0)) {
        JSON_ERROR("the number of Node must at least 1");
    }

    if (graph->nodes) {
        if (unlikely(graph->num_nodes != node_size)) {
            SET_ERROR_RETURN(-1, "the number of nodes is inconsistent");
        }
    } else {
        graph->nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNode) * node_size);
        memset(graph->nodes, 0, sizeof(GraphExecutorNode) * node_size);
    }

    for (uint32_t nid = 0; nid < node_size; ++nid) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "nodes array len expect %zu, parse fail", node_size);

        GraphExecutorNode *node = graph->nodes + nid;
        char key[GRAPH_JSON_KEY_SIZE];
        while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
            if (!strcmp(key, "op")) {
                int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("Parse string for GraphExecutorNode.op fail");
                }

                node->op_type = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                strcpy((char *)node->op_type, global_buf);

            } else if (!strcmp(key, "name")) {
                int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("parse GraphExecutorNode.op fail");
                }

                node->name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                strcpy((char *)node->name, global_buf);
            } else if (!strcmp(key, "inputs")) {
                size_t inputs_num;
                status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &inputs_num);
                if (unlikely(status)) {
                    JSON_ERROR("get GraphExecutorNode.inputs length fail");
                }

                if (inputs_num) {
                    node->num_inputs = inputs_num;
                    node->inputs = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * inputs_num);
                    memset(node->inputs, 0, sizeof(GraphExecutorNodeEntry));
                }
                for (uint32_t inputs_count = 0; inputs_count < inputs_num; ++inputs_count) {
                    ARRAY_CHECK_NEXT_EXISTS(reader, "parse NodeEntry Error"); // '[' or ','

                    // node_id
                    ARRAY_CHECK_NEXT_EXISTS(reader, "no element NodeEntry.node_id"); // '['

                    status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &node->inputs[inputs_count].node_id);
                    if (unlikely(status)) {
                        JSON_ERROR("Read uint32 fail for NodeEntry.node_id");
                    }
                    // index
                    ARRAY_CHECK_NEXT_EXISTS(reader, "no element for NodeEntry.index"); // ','

                    status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &node->inputs[inputs_count].index);
                    if (unlikely(status)) {
                        JSON_ERROR("Read uint32 fail for NodeEntry.index");
                    }

                    // version
                    status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);
                    if (likely(status == 1)) {
                        uint32_t version_tmp;
                        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &version_tmp);
                        if (unlikely(status)) {
                            JSON_ERROR("Read uint32 fail for NodeEntry.version");
                        }

                        ARRAY_CHECK_NEXT_NON_EXISTS(reader, "NodeEntry need len = 2 or 3, but given >3");
                    }
                }
                ARRAY_CHECK_NEXT_NON_EXISTS(reader, "inputs len expect %zu, parse fail", inputs_num); // ']'

            } else if (!strcmp(key, "attr") || !strcmp(key, "attrs")) {
                while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
                    int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                    if (unlikely(str_len == -1)) {
                        JSON_ERROR("Parse string for Node Attrs key=%s fail", key);
                    }

                    if (!strcmp(key, "func_name")) {
                        if (unlikely(str_len == -2)) {
                            JSON_ERROR("node.attr func_name cannot be empty");
                        }
                        node->func_name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                        strcpy((char *)node->func_name, global_buf);
                    } else if (!strcmp(key, "num_inputs")) {
                        uint32_t num_inputs_tmp = 0;
                        STR_DIGIT_TO_UINT(global_buf, str_len, num_inputs_tmp);
                        if (unlikely(node->inputs != NULL && num_inputs_tmp != node->num_inputs)) {
                            JSON_ERROR("JsonReader Data error: Node Attrs.num_inputs(%d) != Attrs.inputs.len(%d)",
                                       num_inputs_tmp, node->num_inputs);
                        }

                    } else if (!strcmp(key, "num_outputs")) {
                        STR_DIGIT_TO_UINT(global_buf, str_len, node->num_outputs);
                    } else if (!strcmp(key, "flatten_data")) {
                        STR_DIGIT_TO_UINT(global_buf, str_len, node->flatten_data);
                    }
                    // else unknown attr key name
                }
            } else if (!strcmp(key, "control_deps")) {
                JSON_ERROR("unimplemented key: %s", key);
            } else {
                JSON_ERROR("unsupported key: %s", key);
            }
        }
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "nodes array len expect %zu, parse fail", node_size);
    return 0;
}

/*!
 * \brief load graph input node indices from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *reader, GraphExecutor *graph) {
    size_t input_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &input_size);
    if (unlikely(status)) {
        JSON_ERROR("parse input node indices array length fail");
    }
    if (unlikely(input_size == 0)) {
        JSON_ERROR("the number of graph input nodes must at least 1");
    }

    graph->inputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * input_size);
    memset(graph->inputs_nodes, 0, sizeof(uint32_t) * input_size);
    graph->num_inputs_nodes = input_size;

    for (size_t input_count = 0; input_count < input_size; ++input_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "parse input node array element error"); // '['

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, graph->inputs_nodes + input_count);
        if (unlikely(status)) {
            JSON_ERROR("parse uint32 fail for inputs_nodes");
        }
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "input node array len expect %zu, parse fail", input_size); // ']'
    return 0;
}

/*!
 * \brief load graph output nodeEntry from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *reader, GraphExecutor *graph) {
    size_t entry_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &entry_size);
    if (unlikely(status)) {
        JSON_ERROR("parse input node indices array length fail");
    }
    if (unlikely(entry_size == 0)) {
        JSON_ERROR("the number of Outputs nodeEntry must at least 1");
    }

    graph->outputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * entry_size);
    memset(graph->outputs_nodes, 0, sizeof(GraphExecutorNodeEntry) * entry_size);
    graph->num_outputs = entry_size;

    for (size_t entry_count = 0; entry_count < entry_size; ++entry_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "parse outputs NodeEntry fail"); // '[' or ','
        // node_id
        ARRAY_CHECK_NEXT_EXISTS(reader, "no element for outputs NodeEntry.node_id"); // '['

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &(graph->outputs_nodes[entry_count].node_id));
        if (unlikely(status)) {
            JSON_ERROR("Read uint32 fail for outputs NodeEntry.node_id");
        }
        // index
        ARRAY_CHECK_NEXT_EXISTS(reader, "no element for outputs NodeEntry.index");

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &(graph->outputs_nodes[entry_count].index));
        if (unlikely(status)) {
            JSON_ERROR("Read uint32 fail for outputs NodeEntry.index");
        }

        // version
        status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);
        if (likely(status == 1)) {
            uint32_t version_tmp;
            status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &version_tmp);
            if (unlikely(status)) {
                JSON_ERROR("Read uint32 fail for outputs NodeEntry.version");
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "NodeEntry need len = 2 or 3, but given >3");
        }
    }
    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "NodeEntry array len expect = %zu, parse fail", entry_size); // ']'

    return 0;
}

/*!
 * \brief load graph attributes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphAttrObject(JsonReader *reader, GraphExecutor *graph) {
    int status = 0;
    size_t storage_id_size;
    size_t device_type_size;
    size_t shape_size;
    size_t data_type_size;
    char key[GRAPH_JSON_KEY_SIZE];

    while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
        if (!strcmp(key, "dltype")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "parse graphAttr dltype fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("parse GraphAttr dltype element fail");
            }
            if (unlikely(strcmp(global_buf, "list_str"))) {
                JSON_ERROR("parse GraphAttr dltype element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr dltype no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &data_type_size);
            if (unlikely(status)) {
                JSON_ERROR("parse GraphAttr data_type array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * data_type_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * data_type_size);
                graph->num_data_entry = data_type_size;
            } else {
                if (unlikely(data_type_size != graph->num_data_entry)) {
                    SET_ERROR_RETURN(-1, "parse GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < data_type_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr data_type array element fail");

                str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("parse GraphAttr data_type array element fail");
                }
                status = TVM_RT_WASM_DLDataType_ParseFromString(global_buf, &graph->data_entry[i].dl_tensor.dtype);
                if (unlikely(status)) {
                    return status;
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'

        } else if (!strcmp(key, "storage_id")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "parse graphAttr storage_id fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("parse GraphAttr storage_id element fail");
            }
            if (unlikely(strcmp(global_buf, "list_int"))) {
                JSON_ERROR("parse GraphAttr storage_id element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr storage_id no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &storage_id_size);
            if (unlikely(status)) {
                JSON_ERROR("parse GraphAttr storage_id array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * storage_id_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * storage_id_size);
                graph->num_data_entry = storage_id_size;
            } else {
                if (unlikely(storage_id_size != graph->num_data_entry)) {
                    SET_ERROR_RETURN(-1, "parse GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < storage_id_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr storage_id array element fail");

                status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &graph->data_entry[i].storage_id);
                if (unlikely(status)) {
                    JSON_ERROR("parse GraphAttr storage_id array element fail");
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
        } else if (!strcmp(key, "device_index")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "parse graphAttr device_index fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("parse GraphAttr device_index element fail");
            }
            if (unlikely(strcmp(global_buf, "list_int"))) {
                JSON_ERROR("parse GraphAttr device_index element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr dev_type no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &device_type_size);
            if (unlikely(status)) {
                JSON_ERROR("parse GraphAttr device_index array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * device_type_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * device_type_size);
                graph->num_data_entry = device_type_size;
            } else {
                if (unlikely(device_type_size != graph->num_data_entry)) {
                    SET_ERROR_RETURN(-1, "parse GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < device_type_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr dev_type array element fail");

                status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &graph->data_entry[i].dl_tensor.device.device_type);
                if (unlikely(status)) {
                    JSON_ERROR("parse GraphAttr dev_type array element fail");
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
        } else if (!strcmp(key, "shape")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "parse graphAttr shape fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("parse GraphAttr shape element fail");
            }
            if (unlikely(strcmp(global_buf, "list_shape"))) {
                JSON_ERROR("parse GraphAttr shape element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr shape no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &shape_size);
            if (unlikely(status)) {
                JSON_ERROR("parse GraphAttr shape array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * shape_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * shape_size);
                graph->num_data_entry = shape_size;
            } else {
                if (unlikely(shape_size != graph->num_data_entry)) {
                    SET_ERROR_RETURN(-1, "parse GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < shape_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr shape array length fail");

                size_t ndim;
                status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &ndim);
                if (unlikely(status)) {
                    JSON_ERROR("parse GraphAttr shape.dim element fail");
                }
                graph->data_entry[i].dl_tensor.shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * ndim);
                memset(graph->data_entry[i].dl_tensor.shape, 0, sizeof(int64_t) * ndim);
                graph->data_entry[i].dl_tensor.ndim = (int)ndim;

                for (size_t dim = 0; dim < ndim; ++dim) {
                    ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr shape.dim element fail");
                    status = TVM_RT_WASM_JsonReader_Read_int64(reader, graph->data_entry[i].dl_tensor.shape + dim);
                    if (unlikely(status)) {
                        JSON_ERROR("parse GraphAttr shape.dim (uint64_t) fail");
                    }
                }
                ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character);"); // ']'
        } else {
            JSON_ERROR("unsupported key (%s) for graphAttr", key);
        }
    }
    if (device_type_size == 0) {
        for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
            graph->data_entry[i].dl_tensor.device = graph->devices[0];
        }
    }

    return status;
}

/*!
 * \brief load graph node_row_ptr from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(JsonReader *reader, GraphExecutor *graph) {
    size_t ptr_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &ptr_size);
    if (unlikely(status)) {
        JSON_ERROR("parse node_row_ptr array length fail");
    }
    if (unlikely(ptr_size == 0)) {
        JSON_ERROR("the number of node_row_ptr must at least 1");
    }

    --ptr_size;

    if (graph->nodes) {
        if (unlikely(graph->num_nodes != ptr_size)) {
            SET_ERROR_RETURN(-1, "the number of nodes and the number of node_row_ptrs are inconsistent");
        }
    } else {
        graph->nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNode) * ptr_size);
        memset(graph->nodes, 0, sizeof(GraphExecutorNode) * ptr_size);
    }

    for (size_t ptr_count = 0; ptr_count < ptr_size; ++ptr_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "parse node_row_ptr array element fail");

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &graph->nodes[ptr_count].row_ptr);
        if (unlikely(status)) {
            JSON_ERROR("parse uint32 Error for node_row_ptr");
        }
    }

    ARRAY_CHECK_NEXT_EXISTS(reader, "parse node_row_ptr array element fail");
    uint32_t tmp; // tmp == graph.num_data_entry
    status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &tmp);
    if (unlikely(status)) {
        JSON_ERROR("parse uint32 Error for node_row_ptr");
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "node_row_ptr len expect %zu", ptr_size);
    return 0;
}
