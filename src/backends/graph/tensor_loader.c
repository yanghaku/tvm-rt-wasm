/*!
 * @file graph/tensor_loader.c
 * @brief load the DLTensor list from param blob.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#include <string.h>

#include <device/cpu_memory.h>
#include <graph/graph_executor.h>
#include <graph/tensor_loader.h>
#include <utils/common.h>

/*!
 * @brief Parse string to DLDataType.
 * @param str the source string
 * @param out_type the pointer to save result DLDataType
 * @return 0 if successful
 */
int TVM_RT_WASM_DLDataType_ParseFromString(const char *str, DLDataType *out_type) {
    if (*str == 0) { // void
        out_type->code = kDLOpaqueHandle;
        out_type->lanes = 0;
        out_type->bits = 0;
        return 0;
    }
    out_type->lanes = 1;
    if (!memcmp(str, "int", 3)) {
        out_type->code = kDLInt;
        out_type->bits = 32;
        str += 3;
    } else if (!memcmp(str, "uint", 4) || !memcmp(str, "bool", 4)) {
        out_type->code = kDLUInt;
        out_type->bits = 32;
        str += 4;
    } else if (!memcmp(str, "float", 5)) {
        out_type->code = kDLFloat;
        out_type->bits = 32;
        str += 4;
    } else if (!memcmp(str, "handle", 6)) {
        out_type->code = kDLOpaqueHandle;
        out_type->bits = 64;
        str += 6;
    } else {
        char *tmp_str = TVM_RT_WASM_WorkplaceMemoryAlloc(strlen(str) + 1);
        strcpy(tmp_str, str);
        TVM_RT_SET_ERROR("Unsupported DLDateType: %s", tmp_str);
        TVM_RT_WASM_WorkplaceMemoryFree(tmp_str);
        return -1;
    }
    if (isdigit1to9(*str)) {
        int num = 0;
        while (isdigit0to9(*str)) {
            (num) = ((num) << 3) + ((num) << 1) + (*str++) - '0';
        }
        out_type->bits = num;
    }
    return 0;
}

/*!
 * @brief Load parameters from stream reader.
 * @param graph The instance of TVM_RT_WASM_GraphExecutor.
 * @param reader The stream reader instance.
 * @return 0 if successful.
 */
int TVM_RT_WASM_GraphExecutorLoadParamsFromReader(TVM_RT_WASM_GraphExecutor graph,
                                                  StreamReader *reader) {
    int status;

    uint64_t tensor_list_magic; // magic(8 bytes)
    status = reader->ReadBytes(reader, &tensor_list_magic, sizeof(uint64_t));
    if (unlikely(status)) {
        return status;
    }
    if (unlikely(tensor_list_magic != kTVMNDArrayListMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Params magic expected %" PRIu64 ", but got %" PRIu64,
                                kTVMNDArrayListMagic, tensor_list_magic);
    }

    // reserved
    status = reader->SkipBytes(reader, sizeof(uint64_t));
    if (unlikely(status)) {
        return status;
    }

    // read std::vector<std::string> name_list;
    uint64_t name_num;
    status = reader->ReadBytes(reader, &name_num, sizeof(uint64_t));
    if (unlikely(status)) {
        return status;
    }
    size_t *name_indexes = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(size_t) * name_num);

    // read names
    for (uint32_t i = 0; i < (uint32_t)name_num; ++i) {
        uint64_t str_len;
        status = reader->ReadBytes(reader, &str_len, sizeof(uint64_t));
        if (unlikely(status)) {
            goto load_param_fail;
        }
        if (unlikely(str_len == 0)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_param_fail, "Node name cannot be empty");
        }

        const char *name = reader->ReadToBuffer(reader, str_len);
        if (unlikely(name == NULL)) {
            goto load_param_fail;
        }

        intptr_t index = -1;
        if (unlikely(TVM_RT_WASM_TrieQueryWithLen(graph->inputs_map, (const uint8_t *)name,
                                                  (size_t)str_len,
                                                  (void **)&index) == TRIE_NOT_FOUND)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_param_fail, "Node name `%s` not found", name);
        }
        name_indexes[i] = (size_t)index;
    }

    uint64_t arr_num;
    status = reader->ReadBytes(reader, &arr_num, sizeof(uint64_t));
    if (unlikely(status)) {
        goto load_param_fail;
    }
    if (unlikely(name_num != arr_num)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_param_fail,
                                  "Params name_num(%" PRIu64 ") != array_num(%" PRIu64 ")",
                                  name_num, arr_num);
    }

    // do load param
    for (uint32_t i = 0; i < (uint32_t)arr_num; ++i) {
        uint32_t eid = DATA_ENTRY_ID(graph, graph->inputs_nodes[name_indexes[i]], 0);
        if (unlikely(eid >= graph->num_data_entry)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(
                load_param_fail, "Data entry id (%u) is greater than the number of data entry (%u)",
                eid, graph->num_data_entry);
        }

        status = TVM_RT_WASM_DLTensor_LoadFromReader(&graph->data_entry[eid].dl_tensor, reader);
        if (unlikely(status)) {
            goto load_param_fail;
        }
    }

    return TVMSynchronize(graph->devices[0].device_type, graph->devices[0].device_id, NULL);

load_param_fail:
    if (name_indexes) {
        TVM_RT_WASM_WorkplaceMemoryFree(name_indexes);
    }
    return status;
}
