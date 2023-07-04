/*!
 * @file relay_vm/relay_instruction.h
 * @brief struct and functions for relay VM function and instructions.
 * @author YangBo MG21330067@smail.nju.edu.cn
 * @sa https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/bytecode.h
 * @note The struct and enum are originally developed by Apache TVM. Apache 2.0 LICENSE.
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_INSTRUCTION_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_INSTRUCTION_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>
#include <utils/stream_reader.h>
#include <utils/trie.h>

/*! @brief Relay VM instruction */
typedef struct TVM_RT_WASM_RelayInstruction_st TVM_RT_WASM_RelayInstruction;

/*! @brief Relay VM function */
typedef struct TVM_RT_WASM_RelayFunction_st *TVM_RT_WASM_RelayFunction;

/*! @brief Relay VM function definition */
struct TVM_RT_WASM_RelayFunction_st {
    /*! @brief function name. */
    char *name;
    /*! @brief function parameter names. */
    char **param_names;
    /*! @brief function parameters device index. */
    size_t *param_device_indices;
    /*! @brief function instructions. */
    TVM_RT_WASM_RelayInstruction *instructions;

    /*! @brief the frame for this function. */
    size_t register_file_size;
    /*! @brief the number of params. */
    size_t num_params;
    /*! @brief the number of instructions. */
    size_t num_instructions;
};

/**
 * @brief Load the rely VM function from stream reader.
 * @param reader The stream reader instance.
 * @param func_ptr The pointer to receive TVM_RT_WASM_RelayFunction.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayFunctionCreateFromReader(StreamReader *reader,
                                              TVM_RT_WASM_RelayFunction *func_ptr);

/*!
 * @brief Free the instance of TVM_RT_WASM_RelayFunction.
 * @param func The instance of TVM_RT_WASM_RelayFunction.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayFunctionFree(TVM_RT_WASM_RelayFunction func);

/*! @brief Relay's opcodes */
enum RelayOpcode {
    RelayOp_Move = 0U,
    RelayOp_Ret = 1U,
    RelayOp_Invoke = 2U,
    RelayOp_InvokeClosure = 3U,
    RelayOp_InvokePacked = 4U,
    RelayOp_AllocTensor = 5U,
    RelayOp_AllocTensorReg = 6U,
    RelayOp_AllocADT = 7U,
    RelayOp_AllocClosure = 8U,
    RelayOp_GetField = 9U,
    RelayOp_If = 10U,
    RelayOp_LoadConst = 11U,
    RelayOp_Goto = 12U,
    RelayOp_GetTag = 13U,
    RelayOp_LoadConstIndex = 14U,
    RelayOp_Fatal = 15U,
    RelayOp_AllocStorage = 16U,
    RelayOp_ShapeOf = 17U,
    RelayOp_ReshapeTensor = 18U,
    RelayOp_DeviceCopy = 19U,
    RelayOp_KillRegister = 20U,
};

/*!
 * @brief Relay VM instruction definition
 * @sa https://github.com/apache/tvm/blob/main/include/tvm/runtime/vm/bytecode.h
 */
struct TVM_RT_WASM_RelayInstruction_st {
    /*! @brief The instruction opcode. */
    enum RelayOpcode op;

    /*! @brief The destination register. */
    size_t reg_dst;

    union {
        struct /* Move Operands */ {
            /*! @brief The source register for a move operation. */
            size_t reg_from;
        } op_move;
        struct /* Return Operands */ {
            /*! @brief The register to return. */
            size_t reg_result;
        } op_ret;
        struct /* Invoke Operands */ {
            /*! @brief The registers containing the arguments. */
            size_t *reg_invoke_args;
            /*! @brief The function to call. */
            size_t func_index;
            /*! @brief The number of arguments to the function. */
            size_t num_args;
        } op_invoke;
        struct /* InvokeClosure Operands */ {
            /*! @brief The register containing the closure. */
            size_t reg_closure;
            /*! @brief The closure arguments as an array. */
            size_t *reg_closure_args;
            /*! @brief The number of arguments to the closure. */
            size_t num_closure;
        } op_invoke_closure;
        struct /* InvokePacked Operands */ {
            /*! @brief The index into the packed function table. */
            size_t packed_index;
            /*! @brief The arity of the packed function. */
            size_t arity;
            /*! @brief The number of outputs produced by the packed function. */
            size_t output_size;
            /*! @brief The arguments to pass to the packed function. */
            size_t *reg_packed_args;
        } op_invoke_packed;
        struct /* AllocTensor Operands */ {
            /*! @brief The storage to allocate from. */
            size_t reg_storage;
            /*! @brief The offset into the storage to allocate from. */
            size_t reg_offset;
            /*! @brief The shape of tensor. */
            int64_t *shape;
            /*! @brief The number of dimensions. */
            size_t ndim;
            /*! @brief The datatype of tensor to be allocated. */
            DLDataType dtype;
        } op_alloc_tensor;
        struct /* AllocTensorReg Operands */ {
            /*! @brief The storage to allocate from. */
            size_t reg_storage;
            /*! @brief The offset into the storage to allocate from. */
            size_t offset;
            /*! @brief The register to read the shape out of. */
            size_t reg_shape;
            /*! @brief The datatype of tensor to be allocated. */
            DLDataType dtype;
        } op_alloc_tensor_reg;
        struct /* AllocADT Operands */ {
            /*! @brief The data type's constructor tag. */
            size_t constructor_tag;
            /*! @brief The fields as an array. */
            size_t *reg_datatype_fields;
            /*! @brief The number of fields to store in the datatype. */
            size_t num_fields;
        } op_alloc_adt;
        struct /* AllocClosure Operands */ {
            /*! @brief The index into the function table. */
            size_t clo_index;
            /*! @brief The free variables as an array. */
            size_t *reg_free_vars;
            /*! @brief The number of free variables to capture. */
            size_t num_free_var;
        } op_alloc_closure;
        struct /* GetField Operands */ {
            /*! @brief The register to project from. */
            size_t reg_object;
            /*! @brief The field to read out. */
            size_t field_index;
        } op_get_field;
        struct /* If Operands */ {
            /*! @brief The register containing the test value. */
            size_t reg_test;
            /*! @brief The register containing the target value. */
            size_t reg_target;
            /*! @brief The program counter offset for the true branch. */
            size_t true_offset;
            /*! @brief The program counter offset for the false branch. */
            size_t false_offset;
        } op_if;
        struct /* LoadConst Operands */ {
            /*! @brief The index into the constant pool. */
            size_t const_index;
        } op_load_const;
        struct /* Jump Operands */ {
            /*! @brief The jump offset. */
            size_t pc_offset;
        } op_goto;
        struct /* GetTag Operands */ {
            /*! @brief The register to project from. */
            size_t reg_object;
        } op_get_tag;
        struct /* LoadConstIndex Operands */ {
            /*! @brief The index into the constant pool. */
            size_t val;
        } op_load_const_index;
        struct /* AllocStorage Operands */ {
            /*! @brief The size of the allocation. */
            size_t reg_allocation_size;
            /*! @brief The alignment of the allocation. */
            size_t alignment;
            /*! @brief The hint of the dtype. */
            DLDataType dtype_hint;
            /*! @brief The index of the device on which the allocation will be made. */
            size_t device_index;
        } op_alloc_storage;
        struct /* ShapeOf Operands */ {
            size_t reg_tensor;
        } op_shape_of;
        struct /* ReshapeTensor Operands */ {
            size_t reg_tensor;
            size_t reg_new_shape;
        } op_reshape_tensor;
        struct /* DeviceCopy Operands */ {
            size_t reg_src;
            /*! @brief The index of the source device to copy from. */
            size_t src_device_index;
            /*! @brief The index of the destination device to copy to. */
            size_t dst_device_index;
        } op_device_copy;
    };
};

/**
 * @brief Load the rely VM instruction from stream reader.
 * @param reader The stream reader instance.
 * @param inst_ptr The pointer to receive TVM_RT_WASM_RelayInstruction.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayInstructionCreateFromReader(StreamReader *reader,
                                                 TVM_RT_WASM_RelayInstruction *inst_ptr);

/*!
 * @brief Free the instance of TVM_RT_WASM_RelayInstruction.
 * @param inst The pointer of TVM_RT_WASM_RelayInstruction.
 * @return 0 if successful.
 */
int TVM_RT_WASM_RelayInstructionFree(TVM_RT_WASM_RelayInstruction *inst);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAY_VM_RELAY_INSTRUCTION_H_INCLUDE_
