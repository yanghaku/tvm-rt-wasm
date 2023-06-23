/*!
 * \file graph/graph_executor.h
 * \brief graph_executor struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_GRAPH_EXECUTOR_INNER_H
#define TVM_RT_WASM_GRAPH_EXECUTOR_INNER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <graph_executor.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/trie.h>

/**
 * A graph.json struct is:
 * {
 *      nodes: [],      // graph nodes information  (array)
 *      arg_nodes: [],  // the node_id for input nodes  (array)
 *      heads: [],      // the node_id for output nodes (array)
 *      attrs: {},      // the graph attributes     (dict)
 *      node_row_ptr: []// the data entry index for node    (array)
 *  }
 *
 *  for every node is:
 *  {
 *      op:  str,       // operator type "null", "tvm_op",...  (str)
 *      name: str,      // node name            (str)
 *      inputs: []      // the { node_id, index, version } tuple list  (array)
 *      attrs: {}       // the node attributes, such as func_name      (dict)
 *  }
 *
 *  for attrs, it mainly has 4 keys:
 *  {
 *      storage_id: {}  // the storage id for every node
 *      shape:      {}  // the data shape for every node
 *      dtypes:     {}  // the data type for every node
 *      device_index: {} // the device id for every node
 *  }
 *
 */

/*! \brief NodeEntry for graph */
typedef struct GraphExecutorNodeEntry {
    /*! \brief id in node list */
    uint32_t node_id;
    /*! \brief an entry that represents output data from a node */
    uint32_t index;
    // /*!\brief the version will not be used in this project */
    // uint32_t version;
} GraphExecutorNodeEntry;

/*! \brief Node for graph */
typedef struct GraphExecutorNode {
    /*! \brief inputs number in attr for node */
    uint32_t num_inputs;
    /*! \brief outputs number in attr for node */
    uint32_t num_outputs;
    /*! \brief flatten_data in attr_for node */
    uint32_t flatten_data;
    /*! \brief node_row_ptr (to quickly get data entry id) */
    uint32_t row_ptr;

    /*! \brief the operator type for node */
    const char *op_type;
    /*! \brief the name for node */
    const char *name;
    /*! \brief the function name in attr for node */
    const char *func_name;
    /*! \brief the inputs data NodeEntry */
    GraphExecutorNodeEntry *inputs;
    // /*! \brief control_dep, this will not be used in this project */
    // uint32_t *control_dep;
} GraphExecutorNode;

/*! \brief operator function information for every node */
typedef struct GraphExecutorNodeOp {
    /*! \brief the number of argument */
    int num_args;
    /*! \brief the return typeCode */
    int return_type_code;
    /*! \brief the return value */
    TVMValue return_value;
    /*! \brief argument type_codes */
    int *arg_type_codes;
    /*! \brief argument values */
    TVMValue *arg_values;
    /*! \brief the function handle */
    TVMFunctionHandle exec;
} GraphExecutorNodeOp;

/*! \brief the data entry */
typedef struct DataEntry {
    DLTensor dl_tensor;
    uint32_t storage_id;
} DataEntry;

/*! \brief the data storage pool entry */
typedef struct StorageEntry {
    void *storage;
    int is_linked_param;
} StorageEntry;

/*! \brief the GraphExecutor struct */
struct TVM_RT_WASM_GraphExecutor_st {
    /*! \brief the number of nodes */
    uint32_t num_nodes;
    /*! \brief the number of input nodes */
    uint32_t num_inputs_nodes;
    /*! \brief the number of outputs node entry */
    uint32_t num_outputs;
    /*! \brief the number of data entry */
    uint32_t num_data_entry;
    /*! \brief the number of device */
    uint32_t num_device;
    /*! \brief Node array */
    GraphExecutorNode *nodes;
    /*! \brief nodeOps array */
    GraphExecutorNodeOp *nodeOps;
    /*! \brief inputs nodes index array */
    uint32_t *inputs_nodes;
    /*! \brief outputs node entry array */
    GraphExecutorNodeEntry *outputs_nodes;
    /*! \brief data_entry array */
    DataEntry *data_entry;
    /*! \brief storage array */
    StorageEntry *storages;
    /*! \brief device array */
    DLDevice *devices;
    /*! \brief module handle */
    TVMModuleHandle module_handle;
    /*! \brief map outputs name to output indices */
    Trie *outputs_map;
    /*! \brief map inputs name to inputs indices */
    Trie *inputs_map;
    /*! \brief the node operator argument value storage pool */
    TVMValue *node_op_arg_value_storage;
    /*! \brief the node operator argument type storage pool */
    int *node_op_arg_type_storage;

    /*!
     * \brief Execute the graph.
     * \param g The instance of This.
     * \return 0 if success
     */
    int (*Run)(struct TVM_RT_WASM_GraphExecutor_st *g);

    /*!
     * \brief Free the extension_data.
     * \param extension_data The pointer to extension_data.
     * \return 0 if successful
     */
    int (*Free)(void *extension_data);

    /*!
     * \brief Clone the extension_data.
     * \param extension_data The pointer to extension_data.
     * \param cloned_extension_data Pointer which receive the new instance.
     * \return 0 if successful
     */
    int (*Clone)(void *extension_data, void **cloned_extension_data);

    /*! \brief for extension */
    void *extension_data;
};

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
                                  uint32_t num_dev, TVM_RT_WASM_GraphExecutor graph);

/*--------------------------------some definition for graph executor function-----------------------------------------*/

#define CHECK_GraphExecutor(g) CHECK_INPUT_POINTER(g, -2, "GraphExecutor")
#define CHECK_NodeRange(max_r, index)                                                                                  \
    do {                                                                                                               \
        if (unlikely((index) >= (max_r))) {                                                                            \
            TVM_RT_SET_ERROR_RETURN(-2, "Invalid argument: expect index in range [0,%d), but got %d", (max_r),         \
                                    (index));                                                                          \
        }                                                                                                              \
    } while (0)

/*! \brief GetEntryId for GraphExecutor */
#define DATA_ENTRY_ID(graph, nid, index) ((graph)->nodes[(nid)].row_ptr + (index))

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_GRAPH_EXECUTOR_INNER_H
