/*!
 * \file runtime/graph/graph_executor.h
 * \brief graph_executor struct definition
 * \author YangBo MG21330067@smail.nju.edu.cn
 *
 */

#ifndef TVM_RT_WASM_GRAPH_EXECUTOR_H
#define TVM_RT_WASM_GRAPH_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/graph_executor_manager.h>
#include <tvm/runtime/utils/trie.h>

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
    /*!\brief the version will not be used in this project */
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

    /*! \brief the operator type for node */
    const char *op_type;
    /*! \brief the name for node */
    const char *name;
    /*! \brief the function name in attr for node */
    const char *func_name;
    /*! \brief the inputs data NodeEntry */
    GraphExecutorNodeEntry *inputs;
    /*! \brief control_dep, this will not be used in this project */
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

/*! \brief the attributes of graph */
typedef struct GraphAttr {
    /*! \brief the number of dataEntry, for validate */
    uint32_t num_entry;
    /*! \brief storage id for every dataEntry */
    uint32_t *storage_id;
    /*! \brief device type for every dataEntry */
    uint32_t *device_type;
    /*! \brief shape for every dataEntry */
    uint64_t **shape;
    /*! \brief ndim for every shape */
    uint32_t *ndim;
    /*! \brief DLTensor type for every dataEntry */
    DLDataType *data_type;
} GraphAttr;

#define GRAPH_BASE_MEMBER                                                                                              \
    /*! \brief the number of nodes */                                                                                  \
    uint32_t num_nodes;                                                                                                \
    /*! \brief the number of input nodes */                                                                            \
    uint32_t num_inputs_nodes;                                                                                         \
    /*! \brief the number of outputs node entry */                                                                     \
    uint32_t num_outputs;                                                                                              \
    /*! \brief the number of node_row_ptr */                                                                           \
    uint32_t num_node_row_ptr;                                                                                         \
    /*! \brief the number of data entry */                                                                             \
    uint32_t num_data_entry;                                                                                           \
    /*! \brief the number of storage */                                                                                \
    uint32_t num_storage;                                                                                              \
    /*! \brief the number of device */                                                                                 \
    uint32_t num_device;                                                                                               \
    /*! \brief Node array */                                                                                           \
    GraphExecutorNode *nodes;                                                                                          \
    /*! \brief nodeOps array */                                                                                        \
    GraphExecutorNodeOp *nodeOps;                                                                                      \
    /*! \brief inputs nodes index array */                                                                             \
    uint32_t *inputs_nodes;                                                                                            \
    /*! \brief outputs node entry array */                                                                             \
    GraphExecutorNodeEntry *outputs_nodes;                                                                             \
    /*! \brief node_row_ptr array (to quickly get data entry id) */                                                    \
    uint32_t *node_row_ptr;                                                                                            \
    /*! \brief data_entry array */                                                                                     \
    DLTensor *data_entry;                                                                                              \
    /*! \brief storage array */                                                                                        \
    void **storages;                                                                                                   \
    /*! \brief device array */                                                                                         \
    DLDevice *devices;                                                                                                 \
    /*! \brief bool flag for storage */                                                                                \
    uint8_t *storage_is_linked_param;                                                                                  \
    /*! \brief module handle */                                                                                        \
    TVMModuleHandle module_handle;                                                                                     \
    /*! \brief map outputs name to output indices */                                                                   \
    Trie *outputs_map;                                                                                                 \
    /*! \brief map inputs name to inputs indices */                                                                    \
    Trie *inputs_map;                                                                                                  \
    /*! \brief graph attributes */                                                                                     \
    GraphAttr graph_attr;

typedef struct GraphExecutor {
    GRAPH_BASE_MEMBER
} GraphExecutor;

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
                                    uint32_t num_dev, GraphExecutorManager **g);

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
                                  uint32_t num_dev, GraphExecutor *graph);

/*!
 * \brief Get total number of nodes.
 * \param g The instance of GraphExecutorManager.
 * \return Total number of nodes.
 */
int TVM_RT_WASM_GraphExecutorGetNumOfNodes(GraphExecutorManager *g);

/*!
 * \brief Get the name of node for given index.
 * \param g The instance of GraphExecutorManager.
 * \param nid the node index
 * \param name the pointer to receive string pointer
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorGetNodeName(GraphExecutorManager *g, uint32_t nid, const char **name);

/*!
 * \brief Get the input index given the name of input.
 * \param g The instance of GraphExecutorManager.
 * \param name The name of the input.
 * \return The index of input.
 */
int TVM_RT_WASM_GraphExecutorGetInputIndex(GraphExecutorManager *g, const char *name);

/*!
 * \brief Get the output index given the name of output.
 * \param g The instance of GraphExecutorManager.
 * \param name The name of the output.
 * \return The index of output.
 */
int TVM_RT_WASM_GraphExecutorGetOutputIndex(GraphExecutorManager *g, const char *name);

/*!
 * \brief get number of input tensors allocated.
 * \param g The instance of GraphExecutorManager.
 * \return integer number of tensors available to use.
 */
int TVM_RT_WASM_GraphExecutorGetNumInputs(GraphExecutorManager *g);

/*!
 * \brief get number of output tensors allocated.
 * \param g The instance of GraphExecutorManager.
 * \return integer number of output tensors allocated.
 */
int TVM_RT_WASM_GraphExecutorGetNumOutputs(GraphExecutorManager *g);

/*!
 * \brief set input to the graph based on name.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param index the index of inputs.
 * \param data_in The input data.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorSetInput(GraphExecutorManager *g, uint32_t index, const DLTensor *data_in);

/*!
 * \brief set input to the graph based on name.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param name the name string for node
 * \param data_in The input data.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorSetInputByName(GraphExecutorManager *g, const char *name, const DLTensor *data_in);

/*!
 * \brief Return NDArray for given output index.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param index The output index.
 * \param out The DLTensor corresponding to given output node index.
 * \return The result of this function execution.
 */
int TVM_RT_WASM_GraphExecutorGetOutput(GraphExecutorManager *g, uint32_t index, DLTensor *data_out);

/*!
 * \brief Load parameters from parameter blob.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \param param_blob A binary blob of parameter.
 * \param param_size The parameter size.
 * \return The result of this function execution.
 */
int TVM_RT_WASM_GraphExecutorLoadParams(GraphExecutorManager *g, const char *param_blob, uint32_t param_size);

/*!
 * \brief Execute the graph.
 * \param g The instance of GraphExecutorManager.
 * \param executor The graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorRun(GraphExecutorManager *g);

/*!
 * \brief Release memory associated with the GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param executor Pointer to graph executor.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorRelease(GraphExecutorManager **g);

/*!
 * \brief Clone a new instance of GraphExecutorManager.
 * \param g The instance of GraphExecutorManager.
 * \param cloned Pointer which receive the new instance.
 * \return 0 if successful
 */
int TVM_RT_WASM_GraphExecutorClone(GraphExecutorManager *g, GraphExecutorManager **cloned);

/*--------------------------------some definition for graph executor function-----------------------------------------*/

#define CHECK_GraphExecutorManager(g)                                                                                  \
    do {                                                                                                               \
        if (unlikely((g) == NULL)) {                                                                                   \
            SET_ERROR_RETURN(-1, "invalid argument for GraphExecutorManager");                                         \
        }                                                                                                              \
        if (unlikely((g)->graphHandle == NULL)) {                                                                      \
            SET_ERROR_RETURN(-1, "GraphExecutor is Not initialized yet!");                                             \
        }                                                                                                              \
    } while (0)

/*! \brief GetEntryId for GraphExecutorManager */
#define DATA_ENTRY_ID(graph, nid, index) ((graph)->node_row_ptr[(nid)] + (index))

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_GRAPH_EXECUTOR_H
