/*!
 * \file graph/graph_executor.c
 * \brief the implement for graph_executor
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <graph/cuda_extension.h>
#include <graph/graph_executor.h>
#include <module/module.h>
#include <string.h>
#include <utils/tensor_helper.h>

TVM_RT_WASM_GraphExecutor TVM_RT_WASM_GraphExecutorCreate(const char *graph_json, TVMModuleHandle module_handle,
                                                          const DLDevice *devices, uint32_t num_dev) {
    // if module_handle is NULL, use the system library.
    if (module_handle == NULL) {
        SET_TIME(t0)
        int status = TVM_RT_WASM_ModuleFactory(MODULE_SYSTEM_LIB, NULL, 0, (Module **)&module_handle);
        if (unlikely(status)) {
            return NULL;
        }
        SET_TIME(t1)
        DURING_PRINT(t1, t0, "sys_lib_create time");
    }
    CHECK_INPUT_POINTER(graph_json, NULL, "Graph Json");
    CHECK_INPUT_POINTER(devices, NULL, "Devices");
    if (unlikely(num_dev == 0)) {
        TVM_RT_SET_ERROR_RETURN(NULL, "Invalid argument: the number of devices cannot be zero, at least 1.");
    }

    SET_TIME(t2)
    // start create graph executor
    TVM_RT_WASM_GraphExecutor graph = TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_GraphExecutor_st));
    memset(graph, 0, sizeof(struct TVM_RT_WASM_GraphExecutor_st));

    int status = TVM_RT_WASM_GraphExecutorLoad(graph_json, module_handle, devices, num_dev, graph);
    if (unlikely(status)) {
        TVM_RT_WASM_GraphExecutorFree(graph);
        return NULL;
    }

    // if cuda graph
    if (devices[0].device_type == kDLCUDA) {
        TVM_RT_WASM_CUDAGraphExecutorExtensionDataCreate(graph);
    }

    // end create graph executor
    SET_TIME(t3)
    DURING_PRINT(t3, t2, "graph build time");
    return graph;
}

int TVM_RT_WASM_GraphExecutorFree(TVM_RT_WASM_GraphExecutor graph) {
    CHECK_GraphExecutor(graph);

    // free nodes
    if (graph->nodes) {
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
    }

    // free node operators
    if (graph->nodeOps) {
        TVM_RT_WASM_HeapMemoryFree(graph->nodeOps);
    }
    if (graph->node_op_arg_type_storage) {
        TVM_RT_WASM_HeapMemoryFree(graph->node_op_arg_type_storage);
    }
    if (graph->node_op_arg_value_storage) {
        TVM_RT_WASM_HeapMemoryFree(graph->node_op_arg_value_storage);
    }

    // free inputs nodes
    if (graph->inputs_nodes) {
        TVM_RT_WASM_HeapMemoryFree(graph->inputs_nodes);
    }

    // free output nodes entry
    if (graph->outputs_nodes) {
        TVM_RT_WASM_HeapMemoryFree(graph->outputs_nodes);
    }

    // free data entry + storage + storage_is_linked_param
    if (graph->storages) {
        for (uint32_t eid = 0; eid < graph->num_data_entry; ++eid) {
            uint32_t sid = graph->data_entry[eid].storage_id;
            if (!graph->storages[sid].is_linked_param) {
                void *ptr = graph->storages[sid].storage;
                if (ptr) {
                    TVMDeviceFreeDataSpace(graph->data_entry[eid].dl_tensor.device, ptr);
                }
                graph->storages[sid].is_linked_param = 1;
            }
            int64_t *shape = graph->data_entry[eid].dl_tensor.shape;
            if (shape) {
                TVM_RT_WASM_HeapMemoryFree(shape);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(graph->storages);
    }
    if (graph->num_data_entry) {
        TVM_RT_WASM_HeapMemoryFree(graph->data_entry);
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

    // free extension data
    if (graph->Free) {
        graph->Free(graph->extension_data);
    }

    // free itself
    TVM_RT_WASM_HeapMemoryFree(graph);
    return 0;
}

int TVM_RT_WASM_GraphExecutorRun(TVM_RT_WASM_GraphExecutor g) {
    CHECK_GraphExecutor(g);

    // if graph has custom run function, such as cuda
    if (g->Run) {
        return g->Run(g);
    }

    int status;
    for (uint32_t i = 0; i < g->num_nodes; ++i) {
        PackedFunction *pf = g->nodeOps[i].exec;
        if (pf) { // call function handle
            status = pf->exec(g->nodeOps[i].arg_values, g->nodeOps[i].arg_type_codes, g->nodeOps[i].num_args,
                              &g->nodeOps[i].return_value, &g->nodeOps[i].return_type_code, pf);
            if (unlikely(status)) {
                return status;
            }
        }
    }
    return 0;
}

int TVM_RT_WASM_GraphExecutorSetInput(TVM_RT_WASM_GraphExecutor g, uint32_t index, const DLTensor *data_in) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    CHECK_NodeRange(g->num_inputs_nodes, index);
    uint32_t eid = DATA_ENTRY_ID(g, g->inputs_nodes[index], 0);
    return TVMDeviceCopyDataFromTo((DLTensor *)data_in, &g->data_entry[eid].dl_tensor, NULL);
}

int TVM_RT_WASM_GraphExecutorSetInputByName(TVM_RT_WASM_GraphExecutor g, const char *name, const DLTensor *data_in) {
    CHECK_INPUT_POINTER(data_in, -2, "DLTensor");
    int index = TVM_RT_WASM_GraphExecutorGetInputIndex(g, name);
    if (unlikely(index == -1)) {
        return index;
    }
    return TVM_RT_WASM_GraphExecutorSetInput(g, index, data_in);
}

int TVM_RT_WASM_GraphExecutorGetOutput(TVM_RT_WASM_GraphExecutor g, uint32_t index, DLTensor *data_out) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    CHECK_NodeRange(g->num_outputs, index);

    uint32_t eid = DATA_ENTRY_ID(g, g->outputs_nodes[index].node_id, g->outputs_nodes[index].index);
    return TVMDeviceCopyDataFromTo(&g->data_entry[eid].dl_tensor, data_out, NULL);
}

int TVM_RT_WASM_GraphExecutorGetOutputByName(TVM_RT_WASM_GraphExecutor g, const char *name, DLTensor *data_out) {
    CHECK_INPUT_POINTER(data_out, -2, "DLTensor");
    int index = TVM_RT_WASM_GraphExecutorGetInputIndex(g, name);
    if (unlikely(index == -1)) {
        return index;
    }
    return TVM_RT_WASM_GraphExecutorGetOutput(g, index, data_out);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param graph The instance of TVM_RT_WASM_GraphExecutor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorLoadParams(TVM_RT_WASM_GraphExecutor graph, const char *param_blob, uint32_t param_size) {
    CHECK_GraphExecutor(graph);
    CHECK_INPUT_POINTER(param_blob, -2, "Param blob");

    if (unlikely(param_size < sizeof(uint64_t) * 2)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Param size is too short, at least %zu", sizeof(uint64_t) * 2);
    }
    if (unlikely(*((uint64_t *)param_blob) != kTVMNDArrayListMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Param magic expected %" PRIu64 ", but got %" PRIu64, kTVMNDArrayListMagic,
                                *((uint64_t *)param_blob));
    }
    const char *blob = param_blob + sizeof(uint64_t) + sizeof(uint64_t); // magic(8 bytes), reserved(8 bytes)

    uint64_t name_num;
    memcpy(&name_num, blob, sizeof(name_num));
    blob += sizeof(name_num);
    const char *name = blob;

    // scan names
    for (uint32_t i = 0; i < (uint32_t)name_num; ++i) {
        uint32_t str_len = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t) + str_len;
        if (unlikely(str_len == 0)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Node name cannot be empty");
        }
    }

    uint64_t arr_num;
    memcpy(&arr_num, blob, sizeof(arr_num));
    blob += sizeof(arr_num);

    if (unlikely(name_num != arr_num)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Params name_num(%" PRIu64 ") != array_num(%" PRIu64 ")", name_num, arr_num);
    }

    // scan name and load param
    for (uint32_t i = 0; i < (uint32_t)arr_num; ++i) {
        uint32_t str_len = (uint32_t) * (uint64_t *)name;
        name += sizeof(uint64_t);

        intptr_t index = -1;
        if (unlikely(TVM_RT_WASM_TrieQueryWithLen(graph->inputs_map, (const uint8_t *)name, str_len, (void **)&index) ==
                     TRIE_NOT_FOUND)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Node name `%s` not found", name);
        }

        uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[index], 0);
        if (unlikely(eid >= graph->num_data_entry)) {
            TVM_RT_SET_ERROR_RETURN(-1, "Data entry id (%u) is greater than the number of data entry (%u)", eid,
                                    graph->num_data_entry);
        }

        int status = TVM_RT_WASM_DLTensor_LoadDataFromBinary(&graph->data_entry[eid].dl_tensor, &blob);
        if (unlikely(status)) {
            return status;
        }

        // point to next name
        name += str_len;
    }

    return TVMSynchronize(graph->devices[0].device_type, graph->devices[0].device_id, NULL);
}

/*!
 * \brief Load parameters from parameter file.
 * \param graph The instance of TVM_RT_WASM_GraphExecutor.
 * \param filename File path to read and load.
 * \return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorLoadParamsFromFile(TVM_RT_WASM_GraphExecutor graph, const char *filename) {
    CHECK_GraphExecutor(graph);
    CHECK_INPUT_POINTER(filename, -2, "Filename");

    FILE *fp = NULL;
    if (unlikely((fp = fopen(filename, "rb")) == NULL)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Cannot open file `%s`", filename);
    }

#define read_from_fp(ptr, len, fp, err_handle_block)                                                                   \
    do {                                                                                                               \
        if (unlikely(fread((ptr), 1, (len), fp) != (len))) {                                                           \
            {                                                                                                          \
                err_handle_block;                                                                                      \
            }                                                                                                          \
            TVM_RT_SET_ERROR_RETURN(-1, "Invalid param file: unexpected EOF");                                         \
        }                                                                                                              \
    } while (0)

    // magic(8 bytes)
    uint64_t array_list_magic;
    read_from_fp(&array_list_magic, sizeof(uint64_t), fp, fclose(fp));
    if (unlikely(array_list_magic != kTVMNDArrayListMagic)) {
        fclose(fp);
        TVM_RT_SET_ERROR_RETURN(-1, "Param magic expected %" PRIu64 ", but got %" PRIu64, kTVMNDArrayListMagic,
                                array_list_magic);
    }

    // reserved(8 bytes)
    read_from_fp(&array_list_magic, sizeof(uint64_t), fp, fclose(fp));

    uint64_t name_num;
    read_from_fp(&name_num, sizeof(uint64_t), fp, fclose(fp));

#define NAME_BUFFER_SIZE 1024

    size_t *name_indexes = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(size_t) * name_num);
    size_t current_name_buf_size = NAME_BUFFER_SIZE;
    char *name_buf = TVM_RT_WASM_WorkplaceMemoryAlloc(current_name_buf_size);

#define do_before_return()                                                                                             \
    do {                                                                                                               \
        fclose(fp);                                                                                                    \
        TVM_RT_WASM_WorkplaceMemoryFree(name_buf);                                                                     \
        TVM_RT_WASM_WorkplaceMemoryFree(name_indexes);                                                                 \
    } while (0)

    // scan names
    for (uint32_t i = 0; i < (uint32_t)name_num; ++i) {
        uint64_t str_len;
        read_from_fp(&str_len, sizeof(uint64_t), fp, do_before_return());

        if (unlikely(str_len == 0)) {
            do_before_return();
            TVM_RT_SET_ERROR_RETURN(-1, "Node name cannot be empty");
        }
        // realloc
        if (str_len >= current_name_buf_size) {
            current_name_buf_size = str_len + 1;
            TVM_RT_WASM_WorkplaceMemoryFree(name_buf);
            name_buf = TVM_RT_WASM_WorkplaceMemoryAlloc(current_name_buf_size);
        }
        read_from_fp(name_buf, str_len, fp, do_before_return());

        intptr_t index = -1;
        if (unlikely(TVM_RT_WASM_TrieQueryWithLen(graph->inputs_map, (const uint8_t *)name_buf, (size_t)str_len,
                                                  (void **)&index) == TRIE_NOT_FOUND)) {
            TVM_RT_SET_ERROR("Node name `%s` not found", name_buf);
            do_before_return();
            return -1;
        }
        name_indexes[i] = (size_t)index;
    }

    uint64_t arr_num;
    read_from_fp(&arr_num, sizeof(uint64_t), fp, do_before_return());
    if (unlikely(name_num != arr_num)) {
        do_before_return();
        TVM_RT_SET_ERROR_RETURN(-1, "Params name_num(%" PRIu64 ") != array_num(%" PRIu64 ")", name_num, arr_num);
    }

    // do load param
    for (uint32_t i = 0; i < (uint32_t)arr_num; ++i) {
        uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[name_indexes[i]], 0);
        if (unlikely(eid >= graph->num_data_entry)) {
            do_before_return();
            TVM_RT_SET_ERROR_RETURN(-1, "Data entry id (%u) is greater than the number of data entry (%u)", eid,
                                    graph->num_data_entry);
        }

        int status = TVM_RT_WASM_DLTensor_LoadDataFromFile(&graph->data_entry[eid].dl_tensor, fp);
        if (unlikely(status)) {
            do_before_return();
            return status;
        }
    }

    do_before_return();
    return TVMSynchronize(graph->devices[0].device_type, graph->devices[0].device_id, NULL);
#undef do_before_return
#undef read_from_fp
#undef NAME_BUFFER_SIZE
}

int TVM_RT_WASM_GraphExecutorClone(TVM_RT_WASM_GraphExecutor g, TVM_RT_WASM_GraphExecutor *cloned) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(cloned, -2, "Cloned GraphExecutor pointer");

    *cloned = TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_GraphExecutor_st));
    memcpy(*cloned, g, sizeof(struct TVM_RT_WASM_GraphExecutor_st));

    TVM_RT_WASM_GraphExecutor new_g = *cloned;
    TVM_RT_WASM_GraphExecutor old_g = g;

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
            if (unlikely(TVMDeviceAllocDataSpace(new_g->data_entry[eid].dl_tensor.device, tmp_storage_size[sid], 0,
                                                 no_type, (void **)(new_g->storages + sid)))) {
                TVM_RT_WASM_WorkplaceMemoryFree(tmp_storage_size);
                // todo: free the cloned graph
                return -1;
            }
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

    // clone extension data
    if (old_g->Clone) {
        new_g->extension_data = NULL;
        old_g->Clone(old_g->extension_data, &new_g->extension_data);
    }

    return 0;
}

int TVM_RT_WASM_GraphExecutorGetNumOfNodes(TVM_RT_WASM_GraphExecutor g) {
    CHECK_GraphExecutor(g);
    return (int)(g->num_nodes);
}

int TVM_RT_WASM_GraphExecutorGetNodeName(TVM_RT_WASM_GraphExecutor g, uint32_t nid, const char **name) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(name, -2, "Name pointer");
    CHECK_NodeRange(g->num_nodes, nid);
    *name = g->nodes[nid].name;
    return 0;
}

int TVM_RT_WASM_GraphExecutorGetInputIndex(TVM_RT_WASM_GraphExecutor g, const char *name) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(name, -2, "Name");

    intptr_t index = -1;
    if (unlikely(TVM_RT_WASM_TrieQuery(g->inputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Node name `%s` not found", name);
    }
    return (int)index;
}

int TVM_RT_WASM_GraphExecutorGetOutputIndex(TVM_RT_WASM_GraphExecutor g, const char *name) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(name, -2, "Name");

    intptr_t index = -1;
    if (unlikely(TVM_RT_WASM_TrieQuery(g->outputs_map, (const uint8_t *)name, (void **)&index) == TRIE_NOT_FOUND)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Node name `%s` not found", name);
    }
    return (int)index;
}

int TVM_RT_WASM_GraphExecutorGetNumInputs(TVM_RT_WASM_GraphExecutor g) {
    CHECK_GraphExecutor(g);
    return (int)(g->num_inputs_nodes);
}

int TVM_RT_WASM_GraphExecutorGetNumOutputs(TVM_RT_WASM_GraphExecutor g) {
    CHECK_GraphExecutor(g);
    return (int)(g->num_outputs);
}

int TVM_RT_WASM_GraphExecutorGetInputDataType(TVM_RT_WASM_GraphExecutor g, uint32_t index, DLDataType *type_ptr) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(type_ptr, -2, "DLDataType pointer");
    CHECK_NodeRange(g->num_inputs_nodes, index);
    uint32_t eid = DATA_ENTRY_ID(g, g->inputs_nodes[index], 0);
    *type_ptr = g->data_entry[eid].dl_tensor.dtype;
    return 0;
}

int TVM_RT_WASM_GraphExecutorGetOutputDataType(TVM_RT_WASM_GraphExecutor g, uint32_t index, DLDataType *type_ptr) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(type_ptr, -2, "DLDataType pointer");
    CHECK_NodeRange(g->num_outputs, index);
    uint32_t eid = DATA_ENTRY_ID(g, g->outputs_nodes[index].node_id, g->outputs_nodes[index].index);
    *type_ptr = g->data_entry[eid].dl_tensor.dtype;
    return 0;
}

int TVM_RT_WASM_GraphExecutorGetInputShape(TVM_RT_WASM_GraphExecutor g, uint32_t index, const int64_t **shape_ptr,
                                           int32_t *ndim_ptr) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(shape_ptr, -2, "shape pointer");
    CHECK_INPUT_POINTER(ndim_ptr, -2, "ndim pointer");
    CHECK_NodeRange(g->num_inputs_nodes, index);
    uint32_t eid = DATA_ENTRY_ID(g, g->inputs_nodes[index], 0);
    *shape_ptr = g->data_entry[eid].dl_tensor.shape;
    *ndim_ptr = g->data_entry[eid].dl_tensor.ndim;
    return 0;
}

int TVM_RT_WASM_GraphExecutorGetOutputShape(TVM_RT_WASM_GraphExecutor g, uint32_t index, const int64_t **shape_ptr,
                                            int32_t *ndim_ptr) {
    CHECK_GraphExecutor(g);
    CHECK_INPUT_POINTER(shape_ptr, -2, "shape pointer");
    CHECK_INPUT_POINTER(ndim_ptr, -2, "ndim pointer");
    CHECK_NodeRange(g->num_outputs, index);
    uint32_t eid = DATA_ENTRY_ID(g, g->outputs_nodes[index].node_id, g->outputs_nodes[index].index);
    *shape_ptr = g->data_entry[eid].dl_tensor.shape;
    *ndim_ptr = g->data_entry[eid].dl_tensor.ndim;
    return 0;
}

// used for js api
#if USE_WEBGPU && defined(__EMSCRIPTEN__) // USE_WEBGPU = 1 && defined(__EMSCRIPTEN__)

TVM_DLL const char *TVM_RT_WASM_JS_GraphExecutorGetNodeName(const TVM_RT_WASM_GraphExecutor g, uint32_t nid) {
    const char *res = NULL;
    TVM_RT_WASM_GraphExecutorGetNodeName(g, nid, &res);
    return res;
}

#endif // USE_WEBGPU = 1 && defined(__EMSCRIPTEN__)
