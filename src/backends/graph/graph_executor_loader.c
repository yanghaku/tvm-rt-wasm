/*!
 * \file graph/graph_executor_loader.c
 * \brief parse json file and load graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <graph/graph_executor.h>
#include <graph/json.h>
#include <module/module.h>
#include <utils/tensor_helper.h>

static int TVM_RT_WASM_GraphExecutor_SetupStorage(TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_GraphExecutor_SetupOpExecs(TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_JsonReader_ReadGraphNodesArray(JsonReader *, TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *,
                                                                 TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *,
                                                                TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_JsonReader_ReadGraphAttrObject(JsonReader *, TVM_RT_WASM_GraphExecutor);
static int TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(JsonReader *, TVM_RT_WASM_GraphExecutor);

#ifndef GRAPH_JSON_KEY_SIZE
#define GRAPH_JSON_KEY_SIZE 128
#endif // GRAPH_JSON_KEY_SIZE

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
int TVM_RT_WASM_GraphExecutorLoad(const char *graph_json, TVMModuleHandle module_handle,
                                  const DLDevice *devices, uint32_t num_dev,
                                  TVM_RT_WASM_GraphExecutor graph) {
    // Init JsonReader
    JsonReader *reader = NULL;
    TVM_RT_WASM_JsonReader_Create(graph_json, &reader);
    int status = 0;

    char key[GRAPH_JSON_KEY_SIZE];
    // start to load graph
    int bitmask = 0;
    while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
        if (!strcmp(key, "nodes")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphNodesArray(reader, graph);
            if (unlikely(status)) {
                goto load_parse_json_fail;
            }
            bitmask |= 1;
        } else if (!strcmp(key, "arg_nodes")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(reader, graph);
            if (unlikely(status)) {
                goto load_parse_json_fail;
            }
            bitmask |= 2;
        } else if (!strcmp(key, "heads")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(reader, graph);
            if (unlikely(status)) {
                goto load_parse_json_fail;
            }
            bitmask |= 4;
        } else if (!strcmp(key, "attrs")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphAttrObject(reader, graph);
            if (unlikely(status)) {
                goto load_parse_json_fail;
            }
            bitmask |= 8;
        } else if (!strcmp(key, "node_row_ptr")) {
            status = TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(reader, graph);
            if (unlikely(status)) {
                goto load_parse_json_fail;
            }
            bitmask |= 16;
        } else if (!strcmp(key, "metadata")) {
            break;
        } else {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_parse_json_fail, "Unsupported Json key: `%s`", key);
        }
    }
    if (unlikely(bitmask != (1 | 2 | 4 | 8 | 16))) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_parse_json_fail,
                                  "Json needs key: nodes,arg_nodes,heads,attrs,node_row_ptr");
    }

    // release JsonReader
    TVM_RT_WASM_JsonReader_Release(reader);

    // other member init
    graph->devices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDevice) * num_dev);
    memcpy(graph->devices, devices, sizeof(DLDevice) * num_dev);
    graph->num_device = num_dev;
    graph->module_handle = module_handle;

    if (unlikely(graph->num_data_entry == 0)) {
        TVM_RT_SET_ERROR_RETURN(status, "The number of graph data_entry at least 1");
    }

    TVM_RT_WASM_TrieCreate(&graph->inputs_map);
    for (uintptr_t i = 0; i < graph->num_inputs_nodes; ++i) {
        uint32_t nid = graph->inputs_nodes[i];
        status = TVM_RT_WASM_TrieInsert(graph->inputs_map, (const uint8_t *)graph->nodes[nid].name,
                                        (void *)i);
        if (status) {
            TVM_RT_SET_ERROR_RETURN(status, "Insert data to inputs map fail");
        }
    }

    TVM_RT_WASM_TrieCreate(&graph->outputs_map);
    for (uintptr_t i = 0; i < graph->num_outputs; ++i) {
        uint32_t nid = graph->outputs_nodes[i].node_id;
        status = TVM_RT_WASM_TrieInsert(graph->outputs_map, (const uint8_t *)graph->nodes[nid].name,
                                        (void *)i);
        if (status) {
            TVM_RT_SET_ERROR_RETURN(status, "Insert data to outputs map fail");
        }
    }

    // init storage
    status = TVM_RT_WASM_GraphExecutor_SetupStorage(graph);
    if (unlikely(status)) {
        return status;
    }
    // init operators
    return TVM_RT_WASM_GraphExecutor_SetupOpExecs(graph);

load_parse_json_fail:
    if (reader) {
        TVM_RT_WASM_JsonReader_Release(reader);
    }
    return status;
}

/*!
 * \brief setup storage for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutor_SetupStorage(TVM_RT_WASM_GraphExecutor graph) {
    DLDataType no_type = {0, 0, 0};
    size_t *storage_size = NULL;
    DLDevice *storage_device = NULL;
    int status;

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
        size_t now_size = TVM_RT_WASM_DLTensor_GetDataSize(
            graph->data_entry[i].dl_tensor.shape, (int)graph->data_entry[i].dl_tensor.ndim);
        now_size = ((graph->data_entry[i].dl_tensor.dtype.bits *
                         graph->data_entry[i].dl_tensor.dtype.lanes +
                     7U) /
                    8U) *
                   now_size;
        if (unlikely(now_size == 0)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(setup_storage_return, "Shape[%d] cannot contains 0", i);
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
            if (unlikely(storage_device[sid].device_type !=
                         graph->data_entry[i].dl_tensor.device.device_type)) {
                status = -1;
                TVM_RT_SET_ERROR_AND_GOTO(
                    setup_storage_return,
                    "The same storage requires the same device_type, but got %d and %d",
                    storage_device[sid].device_type,
                    graph->data_entry[i].dl_tensor.device.device_type);
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
    Module *graph_lib_mod = (Module *)graph->module_handle;
    PackedFunction *func;
    status = graph_lib_mod->GetFunction(graph_lib_mod, lookup_linked_param_func_name, 1,
                                        (TVMFunctionHandle *)&func);
    if (status == 0) {
        TVMValue arg_val, ret_val;
        int arg_type, ret_type;
        arg_type = kTVMArgInt;
        for (uint32_t i = 0; i < num_storage; ++i) {
            arg_val.v_int64 = i;
            status = func->exec(&arg_val, &arg_type, 1, &ret_val, &ret_type, func);
            if (likely(status == 0 && ret_val.v_handle != NULL)) {
                graph->storages[i].is_linked_param = 1;
                graph->storages[i].storage = ret_val.v_handle;
            }
        }
    }

    // alloc memory for storage
    for (uint32_t i = 0; i < num_storage; ++i) {
        if (graph->storages[i].is_linked_param == 0) {

            if (unlikely(status =
                             TVMDeviceAllocDataSpace(storage_device[i], storage_size[i], 0, no_type,
                                                     &(graph->storages[i].storage)))) {
                goto setup_storage_return;
            }
        }
    }

    // set up the data_entry
    for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
        graph->data_entry[i].dl_tensor.data =
            graph->storages[graph->data_entry[i].storage_id].storage;
    }

setup_storage_return:
    if (storage_device) {
        TVM_RT_WASM_WorkplaceMemoryFree(storage_device);
    }
    if (storage_size) {
        TVM_RT_WASM_WorkplaceMemoryFree(storage_size);
    }
    return status;
}

/*!
 * \brief setup operators for graph executor
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_GraphExecutor_SetupOpExecs(TVM_RT_WASM_GraphExecutor graph) {
    // init memory
    graph->nodeOps = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeOp) * graph->num_nodes);
    memset(graph->nodeOps, 0, sizeof(GraphExecutorNodeOp) * graph->num_nodes);

    uint32_t num_op_storage = 0;
    for (uint32_t nid = 0; nid < graph->num_nodes; ++nid) {
        graph->nodeOps[nid].num_args =
            (int)(graph->nodes[nid].num_inputs + graph->nodes[nid].num_outputs);
        num_op_storage += graph->nodeOps[nid].num_args;
    }
    graph->node_op_arg_type_storage = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int) * num_op_storage);
    graph->node_op_arg_value_storage =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMValue) * num_op_storage);
    TVMValue *alloc_value = graph->node_op_arg_value_storage;
    int *alloc_type = graph->node_op_arg_type_storage;

    Module *graph_lib_mod = (Module *)graph->module_handle;
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
                nodeOp->arg_values[node->num_inputs + i].v_handle =
                    &graph->data_entry[eid].dl_tensor;
                nodeOp->arg_type_codes[node->num_inputs + i] = kTVMDLTensorHandle;
            }

            if (strcmp(node->func_name, "__nop") == 0) {
                nodeOp->exec = NULL;
                continue;
            }

            int status =
                graph_lib_mod->GetFunction(graph_lib_mod, node->func_name, 1, &nodeOp->exec);
            if (unlikely(status)) {
                TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function name `%s`", node->func_name);
            }

        } else if (strcmp(node->op_type, "null") == 0) {
            continue;
        } else {
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported graph node op_type: %s", node->op_type);
        }
    }
    return 0;
}

#define JSON_READER_ERROR_PREFIX "Graph Json parse error: "
#define JSON_ERROR(fmt, ...)                                                                       \
    TVM_RT_SET_ERROR_RETURN(-1, "%s" fmt, JSON_READER_ERROR_PREFIX, ##__VA_ARGS__)

/*! \brief json next array item exist check */
#define ARRAY_CHECK_NEXT_EXISTS(reader, fmt, ...)                                                  \
    do {                                                                                           \
        status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);                                     \
        if (unlikely(status != 1)) {                                                               \
            JSON_ERROR(fmt, ##__VA_ARGS__);                                                        \
        }                                                                                          \
    } while (0)

/*! \brief json next array item no-exist check */
#define ARRAY_CHECK_NEXT_NON_EXISTS(reader, fmt, ...)                                              \
    do {                                                                                           \
        status = TVM_RT_WASM_JsonReader_NextArrayItem(reader);                                     \
        if (unlikely(status != 0)) {                                                               \
            JSON_ERROR(fmt, ##__VA_ARGS__);                                                        \
        }                                                                                          \
    } while (0)

/*! \brief parse the digit string */
#define STR_DIGIT_TO_UINT(str, str_len, num)                                                       \
    do {                                                                                           \
        for (int i = 0; i < (str_len); ++i) {                                                      \
            (num) = ((num) << 3) + ((num) << 1) + (str)[i] - '0';                                  \
        }                                                                                          \
    } while (0)

/*!
 * \brief load graph nodes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphNodesArray(JsonReader *reader,
                                                      TVM_RT_WASM_GraphExecutor graph) {
    size_t node_size = 0;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &node_size);
    if (unlikely(status)) {
        JSON_ERROR("Parse Node array length fail");
    }
    graph->num_nodes = (uint32_t)node_size;
    if (unlikely(node_size == 0)) {
        JSON_ERROR("The number of Node must at least 1");
    }

    if (graph->nodes) {
        if (unlikely(graph->num_nodes != node_size)) {
            JSON_ERROR("The number of nodes is inconsistent");
        }
    } else {
        graph->nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNode) * node_size);
        memset(graph->nodes, 0, sizeof(GraphExecutorNode) * node_size);
    }

    for (uint32_t nid = 0; nid < node_size; ++nid) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "Nodes array len expect %zu", node_size);

        GraphExecutorNode *node = graph->nodes + nid;
        char key[GRAPH_JSON_KEY_SIZE];
        while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
            if (!strcmp(key, "op")) {
                int str_len =
                    TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("Parse string for GraphExecutorNode.op fail");
                }

                node->op_type = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                strcpy((char *)node->op_type, global_buf);

            } else if (!strcmp(key, "name")) {
                int str_len =
                    TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("Parse GraphExecutorNode.op fail");
                }

                node->name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                strcpy((char *)node->name, global_buf);
            } else if (!strcmp(key, "inputs")) {
                size_t inputs_num;
                status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &inputs_num);
                if (unlikely(status)) {
                    JSON_ERROR("Parse GraphExecutorNode.inputs length fail");
                }

                if (inputs_num) {
                    node->num_inputs = inputs_num;
                    node->inputs =
                        TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * inputs_num);
                    memset(node->inputs, 0, sizeof(GraphExecutorNodeEntry));
                }
                for (uint32_t inputs_count = 0; inputs_count < inputs_num; ++inputs_count) {
                    ARRAY_CHECK_NEXT_EXISTS(reader, "Parse NodeEntry fail"); // '[' or ','

                    // node_id
                    ARRAY_CHECK_NEXT_EXISTS(reader, "No element for NodeEntry.node_id"); // '['

                    status = TVM_RT_WASM_JsonReader_Read_uint32(
                        reader, &node->inputs[inputs_count].node_id);
                    if (unlikely(status)) {
                        JSON_ERROR("Read uint32 fail for NodeEntry.node_id");
                    }
                    // index
                    ARRAY_CHECK_NEXT_EXISTS(reader, "No element for NodeEntry.index"); // ','

                    status = TVM_RT_WASM_JsonReader_Read_uint32(reader,
                                                                &node->inputs[inputs_count].index);
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

                        ARRAY_CHECK_NEXT_NON_EXISTS(reader,
                                                    "NodeEntry need len = 2 or 3, but given >3");
                    }
                }
                ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Inputs len expect %zu, parse fail",
                                            inputs_num); // ']'

            } else if (!strcmp(key, "attr") || !strcmp(key, "attrs")) {
                while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) >
                       0) {
                    int str_len =
                        TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                    if (unlikely(str_len == -1)) {
                        JSON_ERROR("Parse string for Node Attrs key fail");
                    }

                    if (!strcmp(key, "func_name")) {
                        if (unlikely(str_len == -2)) {
                            JSON_ERROR("Node.attrs func_name cannot be empty");
                        }
                        node->func_name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
                        strcpy((char *)node->func_name, global_buf);
                    } else if (!strcmp(key, "num_inputs")) {
                        uint32_t num_inputs_tmp = 0;
                        STR_DIGIT_TO_UINT(global_buf, str_len, num_inputs_tmp);
                        if (unlikely(node->inputs != NULL && num_inputs_tmp != node->num_inputs)) {
                            JSON_ERROR("Node Attrs.num_inputs(%d) != Attrs.inputs.len(%d)",
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
                JSON_ERROR("Unimplemented key: %s", key);
            } else {
                JSON_ERROR("Unsupported key: %s", key);
            }
        }
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Nodes array len expect %zu", node_size);
    global_buf[0] = '\0';
    return 0;
}

/*!
 * \brief load graph input node indices from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphInputNodeIndicesArray(JsonReader *reader,
                                                                 TVM_RT_WASM_GraphExecutor graph) {
    size_t input_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &input_size);
    if (unlikely(status)) {
        JSON_ERROR("Parse input node array length fail");
    }
    if (unlikely(input_size == 0)) {
        JSON_ERROR("The number of graph input nodes must at least 1");
    }

    graph->inputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(uint32_t) * input_size);
    memset(graph->inputs_nodes, 0, sizeof(uint32_t) * input_size);
    graph->num_inputs_nodes = input_size;

    for (size_t input_count = 0; input_count < input_size; ++input_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "Parse input node array element error"); // '['

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, graph->inputs_nodes + input_count);
        if (unlikely(status)) {
            JSON_ERROR("Parse uint32 fail for inputs_nodes");
        }
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Input node array len expect %zu", input_size); // ']'
    return 0;
}

/*!
 * \brief load graph output nodeEntry from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphOutputNodeEntryArray(JsonReader *reader,
                                                                TVM_RT_WASM_GraphExecutor graph) {
    size_t entry_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &entry_size);
    if (unlikely(status)) {
        JSON_ERROR("Parse outputs NodeEntry array length fail");
    }
    if (unlikely(entry_size == 0)) {
        JSON_ERROR("The number of outputs NodeEntry must at least 1");
    }

    graph->outputs_nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNodeEntry) * entry_size);
    memset(graph->outputs_nodes, 0, sizeof(GraphExecutorNodeEntry) * entry_size);
    graph->num_outputs = entry_size;

    for (size_t entry_count = 0; entry_count < entry_size; ++entry_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "Parse outputs NodeEntry fail"); // '[' or ','
        // node_id
        ARRAY_CHECK_NEXT_EXISTS(reader, "No element for outputs NodeEntry.node_id"); // '['

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader,
                                                    &(graph->outputs_nodes[entry_count].node_id));
        if (unlikely(status)) {
            JSON_ERROR("Read uint32 fail for outputs NodeEntry.node_id");
        }
        // index
        ARRAY_CHECK_NEXT_EXISTS(reader, "No element for outputs NodeEntry.index");

        status =
            TVM_RT_WASM_JsonReader_Read_uint32(reader, &(graph->outputs_nodes[entry_count].index));
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
    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "NodeEntry array len expect = %zu", entry_size); // ']'
    return 0;
}

/*!
 * \brief load graph attributes from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphAttrObject(JsonReader *reader,
                                                      TVM_RT_WASM_GraphExecutor graph) {
    int status = 0;
    size_t storage_id_size = 0;
    size_t device_type_size = 0;
    size_t shape_size = 0;
    size_t data_type_size = 0;
    char key[GRAPH_JSON_KEY_SIZE];

    while (TVM_RT_WASM_JsonReader_NextObjectItem(reader, key, GRAPH_JSON_KEY_SIZE) > 0) {
        if (!strcmp(key, "dltype")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "Parse graphAttr dltype fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("Parse GraphAttr dltype element fail");
            }
            if (unlikely(strcmp(global_buf, "list_str"))) {
                JSON_ERROR("Parse GraphAttr dltype element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr dltype no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &data_type_size);
            if (unlikely(status)) {
                JSON_ERROR("Parse GraphAttr data_type array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * data_type_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * data_type_size);
                graph->num_data_entry = data_type_size;
            } else {
                if (unlikely(data_type_size != graph->num_data_entry)) {
                    JSON_ERROR("GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < data_type_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr data_type array element fail");

                str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
                if (unlikely(str_len <= 0)) {
                    JSON_ERROR("Parse GraphAttr data_type array element fail");
                }
                status = TVM_RT_WASM_DLDataType_ParseFromString(
                    global_buf, &graph->data_entry[i].dl_tensor.dtype);
                if (unlikely(status)) {
                    return status;
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character);"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character);"); // ']'

        } else if (!strcmp(key, "storage_id")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "Parse graphAttr storage_id fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("Parse GraphAttr storage_id element fail");
            }
            if (unlikely(strcmp(global_buf, "list_int"))) {
                JSON_ERROR("GraphAttr storage_id element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "GraphAttr storage_id no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &storage_id_size);
            if (unlikely(status)) {
                JSON_ERROR("Parse GraphAttr storage_id array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * storage_id_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * storage_id_size);
                graph->num_data_entry = storage_id_size;
            } else {
                if (unlikely(storage_id_size != graph->num_data_entry)) {
                    JSON_ERROR("GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < storage_id_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr storage_id array element fail");

                status =
                    TVM_RT_WASM_JsonReader_Read_uint32(reader, &graph->data_entry[i].storage_id);
                if (unlikely(status)) {
                    JSON_ERROR("Parse GraphAttr storage_id array element fail");
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character");   // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "invalid array end character");   // ']'
        } else if (!strcmp(key, "device_index")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "Parse graphAttr device_index fail"); // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("Parse GraphAttr device_index element fail");
            }
            if (unlikely(strcmp(global_buf, "list_int"))) {
                JSON_ERROR("GraphAttr device_index element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "parse GraphAttr dev_type no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &device_type_size);
            if (unlikely(status)) {
                JSON_ERROR("Parse GraphAttr device_index array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * device_type_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * device_type_size);
                graph->num_data_entry = device_type_size;
            } else {
                if (unlikely(device_type_size != graph->num_data_entry)) {
                    JSON_ERROR("GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < device_type_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr dev_type array element fail");

                status = TVM_RT_WASM_JsonReader_Read_uint32(
                    reader, &graph->data_entry[i].dl_tensor.device.device_type);
                if (unlikely(status)) {
                    JSON_ERROR("Parse GraphAttr dev_type array element fail");
                }
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character"); // ']'
        } else if (!strcmp(key, "shape")) {
            ARRAY_CHECK_NEXT_EXISTS(reader, "Parse graphAttr shape fail");      // '['

            int str_len = TVM_RT_WASM_JsonReader_ReadString(reader, global_buf, GLOBAL_BUF_SIZE);
            if (unlikely(str_len <= 0)) {
                JSON_ERROR("Parse GraphAttr shape element fail");
            }
            if (unlikely(strcmp(global_buf, "list_shape"))) {
                JSON_ERROR("GraphAttr shape element expect list_str");
            }

            ARRAY_CHECK_NEXT_EXISTS(reader, "GraphAttr shape no array entry"); // '['

            status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &shape_size);
            if (unlikely(status)) {
                JSON_ERROR("Parse GraphAttr shape array length fail");
            }
            if (graph->data_entry == NULL) {
                graph->data_entry = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DataEntry) * shape_size);
                memset(graph->data_entry, 0, sizeof(DataEntry) * shape_size);
                graph->num_data_entry = shape_size;
            } else {
                if (unlikely(shape_size != graph->num_data_entry)) {
                    JSON_ERROR("GraphAttr: num_data_entry inconsistent");
                }
            }

            for (size_t i = 0; i < shape_size; ++i) {
                ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr shape array length fail");

                size_t ndim;
                status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &ndim);
                if (unlikely(status)) {
                    JSON_ERROR("Parse GraphAttr shape.dim element fail");
                }
                graph->data_entry[i].dl_tensor.shape =
                    TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * ndim);
                memset(graph->data_entry[i].dl_tensor.shape, 0, sizeof(int64_t) * ndim);
                graph->data_entry[i].dl_tensor.ndim = (int)ndim;

                for (size_t dim = 0; dim < ndim; ++dim) {
                    ARRAY_CHECK_NEXT_EXISTS(reader, "Parse GraphAttr shape.dim element fail");
                    status = TVM_RT_WASM_JsonReader_Read_int64(
                        reader, graph->data_entry[i].dl_tensor.shape + dim);
                    if (unlikely(status)) {
                        JSON_ERROR("Parse GraphAttr shape.dim (uint64_t) fail");
                    }
                }
                ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character"); // ']'
            }

            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character"); // ']'
            ARRAY_CHECK_NEXT_NON_EXISTS(reader, "Invalid array end character"); // ']'
        } else {
            JSON_ERROR("Unsupported key `%s` for GraphAttr", key);
        }
    }
    if (device_type_size == 0) {
        for (uint32_t i = 0; i < graph->num_data_entry; ++i) {
            graph->data_entry[i].dl_tensor.device = graph->devices[0];
        }
    }

    global_buf[0] = '\0';
    return status;
}

/*!
 * \brief load graph node_row_ptr from json
 * @param reader the instance of JsonReader
 * @param graph the instance of GraphExecutor
 * @return 0 if successful
 */
static int TVM_RT_WASM_JsonReader_ReadGraphNodeRowPtrArray(JsonReader *reader,
                                                           TVM_RT_WASM_GraphExecutor graph) {
    size_t ptr_size;
    int status = TVM_RT_WASM_JsonReader_ArrayLength(reader, &ptr_size);
    if (unlikely(status)) {
        JSON_ERROR("Parse node_row_ptr array length fail");
    }
    if (unlikely(ptr_size == 0)) {
        JSON_ERROR("The number of node_row_ptr must at least 1");
    }

    --ptr_size;

    if (graph->nodes) {
        if (unlikely(graph->num_nodes != ptr_size)) {
            JSON_ERROR("The number of nodes is inconsistent");
        }
    } else {
        graph->nodes = TVM_RT_WASM_HeapMemoryAlloc(sizeof(GraphExecutorNode) * ptr_size);
        memset(graph->nodes, 0, sizeof(GraphExecutorNode) * ptr_size);
    }

    for (size_t ptr_count = 0; ptr_count < ptr_size; ++ptr_count) {
        ARRAY_CHECK_NEXT_EXISTS(reader, "Parse node_row_ptr array element fail");

        status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &graph->nodes[ptr_count].row_ptr);
        if (unlikely(status)) {
            JSON_ERROR("Parse uint32 Error for node_row_ptr");
        }
    }

    ARRAY_CHECK_NEXT_EXISTS(reader, "Parse node_row_ptr array element fail");
    uint32_t tmp; // tmp == graph.num_data_entry
    status = TVM_RT_WASM_JsonReader_Read_uint32(reader, &tmp);
    if (unlikely(status)) {
        JSON_ERROR("Parse uint32 Error for node_row_ptr");
    }

    ARRAY_CHECK_NEXT_NON_EXISTS(reader, "node_row_ptr len expect %zu", ptr_size);
    return 0;
}