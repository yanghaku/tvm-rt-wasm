/*!
 * @file relay_vm/relay_instruction.c
 * @brief functions for relay VM function and instructions.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#include <device/cpu_memory.h>
#include <relay_vm/relay_instruction.h>
#include <utils/common.h>

INLINE int64_t parse_str_ull__(const char *s, size_t len) {
    int64_t ans = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = s[i];
        if (!isdigit0to9(c)) {
            return -1;
        }
        ans = (ans << 3) + (ans << 1) + (c - '0');
    }
    return ans;
}

/**! @sa https://github.com/apache/tvm/blob/main/src/support/utils.h */
INLINE uint64_t hash__(uint64_t state, uint64_t value) {
    return state ^ (value + 0x9e3779b9 + (state << 6) + (state >> 2));
}

int TVM_RT_WASM_RelayFunctionCreateFromReader(StreamReader *reader,
                                              TVM_RT_WASM_RelayFunction *func_ptr) {
    TVM_RT_WASM_RelayFunction func =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_RelayFunction_st));
    memset(func, 0, sizeof(struct TVM_RT_WASM_RelayFunction_st));
    int status;

    uint64_t func_info_len;
    if (unlikely(status = reader->ReadBytes(reader, &func_info_len, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    if (unlikely(func_info_len != 3)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_func_fail,
                                  "The VM function info length must be 3 but got `%" PRIu64 "`",
                                  func_info_len);
    }
    uint64_t str_len;
    // read name
    if (unlikely(status = reader->ReadBytes(reader, &str_len, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    func->name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
    func->name[str_len] = 0;
    if (unlikely(status = reader->ReadBytes(reader, func->name, str_len))) {
        goto load_func_fail;
    }

    // read register file size
    if (unlikely(status = reader->ReadBytes(reader, &str_len, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    const char *s = reader->ReadToBuffer(reader, str_len);
    if (unlikely(s == NULL)) {
        goto load_func_fail;
    }
    int64_t l = parse_str_ull__(s, (size_t)str_len);
    if (l < 0) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_func_fail, "Cannot parse string %s", s);
    }
    func->register_file_size = (size_t)l;

    // read number of instructions
    if (unlikely(status = reader->ReadBytes(reader, &str_len, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    s = reader->ReadToBuffer(reader, str_len);
    if (unlikely(s == NULL)) {
        goto load_func_fail;
    }
    l = parse_str_ull__(s, (size_t)str_len);
    if (l < 0) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_func_fail, "Cannot parse string %s", s);
    }
    func->num_instructions = (size_t)l;

    // read params
    uint64_t num_params;
    if (unlikely(status = reader->ReadBytes(reader, &num_params, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    func->num_params = (size_t)num_params;
    func->param_names = TVM_RT_WASM_HeapMemoryAlloc(sizeof(char *) * func->num_params);
    memset(func->param_names, 0, sizeof(char *) * func->num_params);
    for (size_t i = 0; i < func->num_params; ++i) {
        if (unlikely(status = reader->ReadBytes(reader, &str_len, sizeof(uint64_t)))) {
            goto load_func_fail;
        }
        func->param_names[i] = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
        func->param_names[i][str_len] = 0;
        if (unlikely(status = reader->ReadBytes(reader, func->param_names[i], str_len))) {
            goto load_func_fail;
        }
    }

    // read params device index
    uint64_t num_params_validate;
    if (unlikely(status = reader->ReadBytes(reader, &num_params_validate, sizeof(uint64_t)))) {
        goto load_func_fail;
    }
    if (unlikely(num_params != num_params_validate)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(
            load_func_fail, "The number of params device index expect %" PRIu64 " but got %" PRIu64,
            num_params, num_params_validate);
    }
    func->param_device_indices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * num_params);
    const int64_t *ids =
        (const int64_t *)reader->ReadToBuffer(reader, sizeof(int64_t) * func->num_params);
    if (unlikely(ids == NULL)) {
        goto load_func_fail;
    }
    for (size_t i = 0; i < func->num_params; ++i) {
        func->param_device_indices[i] = (size_t)ids[i];
    }

    func->instructions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVM_RT_WASM_RelayInstruction) * func->num_instructions);
    memset(func->instructions, 0, sizeof(TVM_RT_WASM_RelayInstruction) * func->num_instructions);
    // load instructions
    for (size_t i = 0; i < func->num_instructions; ++i) {
        if (unlikely(status = TVM_RT_WASM_RelayInstructionCreateFromReader(
                         reader, func->instructions + i))) {
            goto load_func_fail;
        }
    }

    *func_ptr = func;
    return 0;
load_func_fail:
    TVM_RT_WASM_RelayFunctionFree(func);
    return status;
}

int TVM_RT_WASM_RelayFunctionFree(TVM_RT_WASM_RelayFunction func) {
    if (func->name) {
        TVM_RT_WASM_HeapMemoryFree(func->name);
    }
    if (func->param_names) {
        for (size_t i = 0; i < func->num_params; ++i) {
            if (func->param_names[i]) {
                TVM_RT_WASM_HeapMemoryFree(func->param_names[i]);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(func->param_names);
    }
    if (func->param_device_indices) {
        TVM_RT_WASM_HeapMemoryFree(func->param_device_indices);
    }
    if (func->instructions) {
        for (size_t i = 0; i < func->num_instructions; ++i) {
            TVM_RT_WASM_RelayInstructionFree(func->instructions + i);
        }
        TVM_RT_WASM_HeapMemoryFree(func->instructions);
    }
    TVM_RT_WASM_HeapMemoryFree(func);
    return 0;
}

static int TVM_RT_WASM_RelayInstructionCreateFromFields(TVM_RT_WASM_RelayInstruction *inst_ptr,
                                                        enum RelayOpcode opcode,
                                                        const int64_t *fields, size_t num_fields) {
#define CHECK_FIELD_SIZE(expected, cmp)                                                            \
    do {                                                                                           \
        if (unlikely(num_fields cmp(size_t)(expected))) {                                          \
            TVM_RT_SET_ERROR_RETURN(-1, "Check instruction fields number fail: %zu " #cmp " %zu",  \
                                    (size_t)(expected), num_fields);                               \
        }                                                                                          \
    } while (0)

#define CHECK_FIELD_EQ(expected) CHECK_FIELD_SIZE(expected, !=)
#define CHECK_FIELD_GE(expected) CHECK_FIELD_SIZE(expected, <)

    switch (opcode) {
    case RelayOp_Move:
        CHECK_FIELD_EQ(2);
        inst_ptr->op_move.reg_from = (size_t)fields[0];
        inst_ptr->reg_dst = (size_t)fields[1];
        break;
    case RelayOp_Ret:
        CHECK_FIELD_EQ(1);
        inst_ptr->op_ret.reg_result = (size_t)fields[0];
        break;
    case RelayOp_Invoke: {
        CHECK_FIELD_GE(3);
        size_t num_args = (size_t)fields[1];
        size_t expected = 3 + num_args;
        CHECK_FIELD_EQ(expected);

        inst_ptr->op_invoke.func_index = (size_t)fields[0];
        inst_ptr->op_invoke.num_args = num_args;
        inst_ptr->reg_dst = (size_t)fields[2];
        inst_ptr->op_invoke.reg_invoke_args =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * num_args);
        for (size_t i = 0; i < num_args; ++i) {
            inst_ptr->op_invoke.reg_invoke_args[i] = (size_t)fields[i + 3];
        }
        break;
    }
    case RelayOp_InvokeClosure: {
        CHECK_FIELD_GE(3);
        size_t num_args = (size_t)fields[1];
        size_t expected = 3 + num_args;
        CHECK_FIELD_EQ(expected);

        inst_ptr->op_invoke_closure.reg_closure = (size_t)fields[0];
        inst_ptr->op_invoke_closure.num_closure = num_args;
        inst_ptr->reg_dst = (size_t)fields[2];
        inst_ptr->op_invoke_closure.reg_closure_args =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * num_args);
        for (size_t i = 0; i < num_args; ++i) {
            inst_ptr->op_invoke_closure.reg_closure_args[i] = (size_t)fields[i + 3];
        }
        break;
    }
    case RelayOp_InvokePacked: {
        CHECK_FIELD_GE(3);
        size_t num_packed_args = (size_t)fields[1];
        size_t expected = 3 + num_packed_args;
        CHECK_FIELD_GE(expected);

        inst_ptr->op_invoke_packed.packed_index = (size_t)fields[0];
        inst_ptr->op_invoke_packed.arity = num_packed_args;
        inst_ptr->op_invoke_packed.output_size = (size_t)fields[2];
        inst_ptr->op_invoke_packed.reg_packed_args =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * num_packed_args);
        for (size_t i = 0; i < num_packed_args; ++i) {
            inst_ptr->op_invoke_packed.reg_packed_args[i] = (size_t)fields[i + 3];
        }
        break;
    }
    case RelayOp_AllocTensor: {
        CHECK_FIELD_GE(7);
        size_t ndim = (size_t)fields[5];
        size_t expected = 7 + ndim;
        CHECK_FIELD_EQ(expected);

        inst_ptr->op_alloc_tensor.reg_storage = (size_t)fields[0];
        inst_ptr->op_alloc_tensor.reg_offset = (size_t)fields[1];

        DLDataType dtype;
        dtype.code = (uint8_t)fields[2];
        dtype.bits = (uint8_t)fields[3];
        dtype.lanes = (uint16_t)fields[4];
        inst_ptr->op_alloc_tensor.dtype = dtype;

        inst_ptr->op_alloc_tensor.ndim = ndim;
        inst_ptr->reg_dst = (size_t)fields[6];
        inst_ptr->op_alloc_tensor.shape = TVM_RT_WASM_HeapMemoryAlloc(sizeof(int64_t) * ndim);
        memcpy(inst_ptr->op_alloc_tensor.shape, fields + 7, sizeof(int64_t) * ndim);
        break;
    }
    case RelayOp_AllocTensorReg: {
        CHECK_FIELD_EQ(7);

        inst_ptr->op_alloc_tensor_reg.reg_storage = (size_t)fields[0];
        inst_ptr->op_alloc_tensor_reg.offset = (size_t)fields[1];
        inst_ptr->op_alloc_tensor_reg.reg_shape = (size_t)fields[2];

        DLDataType dtype;
        dtype.code = (uint8_t)fields[3];
        dtype.bits = (uint8_t)fields[4];
        dtype.lanes = (uint16_t)fields[5];
        inst_ptr->op_alloc_tensor_reg.dtype = dtype;

        inst_ptr->reg_dst = (size_t)fields[6];
        break;
    }
    case RelayOp_AllocADT: {
        CHECK_FIELD_GE(3);
        size_t adt_num_fields = (size_t)fields[1];
        size_t expected = 3 + adt_num_fields;
        CHECK_FIELD_EQ(expected);

        inst_ptr->op_alloc_adt.constructor_tag = (size_t)fields[0];
        inst_ptr->op_alloc_adt.num_fields = adt_num_fields;
        inst_ptr->reg_dst = (size_t)fields[2];

        inst_ptr->op_alloc_adt.reg_datatype_fields =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * adt_num_fields);
        for (size_t i = 0; i < adt_num_fields; ++i) {
            inst_ptr->op_alloc_adt.reg_datatype_fields[i] = (size_t)fields[i + 3];
        }
        break;
    }
    case RelayOp_AllocClosure: {
        CHECK_FIELD_GE(3);
        size_t num_free_var = (size_t)fields[1];
        size_t expected = 3 + num_free_var;
        CHECK_FIELD_EQ(expected);

        inst_ptr->op_alloc_closure.clo_index = (size_t)fields[0];
        inst_ptr->op_alloc_closure.num_free_var = num_free_var;
        inst_ptr->reg_dst = (size_t)fields[2];
        inst_ptr->op_alloc_closure.reg_free_vars =
            TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * num_free_var);
        for (size_t i = 0; i < num_free_var; ++i) {
            inst_ptr->op_alloc_closure.reg_free_vars[i] = fields[i + 3];
        }
        break;
    }
    case RelayOp_GetField:
        CHECK_FIELD_EQ(3);
        inst_ptr->op_get_field.reg_object = (size_t)fields[0];
        inst_ptr->op_get_field.field_index = (size_t)fields[1];
        inst_ptr->reg_dst = (size_t)fields[2];
        break;
    case RelayOp_If:
        CHECK_FIELD_EQ(4);
        inst_ptr->op_if.reg_test = (size_t)fields[0];
        inst_ptr->op_if.reg_target = (size_t)fields[1];
        inst_ptr->op_if.true_offset = (size_t)fields[2];
        inst_ptr->op_if.false_offset = (size_t)fields[3];
        break;
    case RelayOp_LoadConst:
        CHECK_FIELD_EQ(2);
        inst_ptr->op_load_const.const_index = (size_t)fields[0];
        inst_ptr->reg_dst = (size_t)fields[1];
        break;
    case RelayOp_Goto:
        CHECK_FIELD_EQ(1);
        inst_ptr->op_goto.pc_offset = (size_t)fields[0];
        break;
    case RelayOp_GetTag:
        CHECK_FIELD_EQ(2);
        inst_ptr->op_get_tag.reg_object = (size_t)fields[0];
        inst_ptr->reg_dst = (size_t)fields[1];
        break;
    case RelayOp_LoadConstIndex:
        CHECK_FIELD_EQ(2);
        inst_ptr->op_load_const_index.val = (size_t)fields[0];
        inst_ptr->reg_dst = (size_t)fields[1];
        break;
    case RelayOp_Fatal:
        CHECK_FIELD_EQ(0);
        break;
    case RelayOp_AllocStorage: {
        CHECK_FIELD_EQ(7);
        inst_ptr->op_alloc_storage.reg_allocation_size = (size_t)fields[0];
        inst_ptr->op_alloc_storage.alignment = (size_t)fields[1];

        DLDataType dtype;
        dtype.code = (uint8_t)fields[2];
        dtype.bits = (uint8_t)fields[3];
        dtype.lanes = (uint16_t)fields[4];
        inst_ptr->op_alloc_storage.dtype_hint = dtype;

        inst_ptr->op_alloc_storage.device_index = (size_t)fields[5];
        inst_ptr->reg_dst = (size_t)fields[6];
        break;
    }
    case RelayOp_ShapeOf:
        CHECK_FIELD_EQ(2);
        inst_ptr->op_shape_of.reg_tensor = (size_t)fields[0];
        inst_ptr->reg_dst = (size_t)fields[1];
        break;
    case RelayOp_ReshapeTensor:
        CHECK_FIELD_EQ(3);
        inst_ptr->op_reshape_tensor.reg_tensor = (size_t)fields[0];
        inst_ptr->op_reshape_tensor.reg_new_shape = (size_t)fields[1];
        inst_ptr->reg_dst = (size_t)fields[2];
        break;
    case RelayOp_DeviceCopy:
        CHECK_FIELD_EQ(4);
        inst_ptr->op_device_copy.reg_src = (size_t)fields[0];
        inst_ptr->op_device_copy.src_device_index = (size_t)fields[1];
        inst_ptr->op_device_copy.dst_device_index = (size_t)fields[2];
        inst_ptr->reg_dst = (size_t)fields[3];
        break;
    case RelayOp_KillRegister:
        CHECK_FIELD_EQ(1);
        inst_ptr->reg_dst = (size_t)fields[0];
        break;
    default:
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid Relay Opcode %u", opcode);
    }

    inst_ptr->op = opcode;
    return 0;
#undef CHECK_FIELD_EQ
#undef CHECK_FIELD_GE
#undef CHECK_FIELD_SIZE
}

int TVM_RT_WASM_RelayInstructionCreateFromReader(StreamReader *reader,
                                                 TVM_RT_WASM_RelayInstruction *inst_ptr) {
    uint64_t inst_size;
    int status;
    if (unlikely(status = reader->ReadBytes(reader, &inst_size, sizeof(uint64_t)))) {
        return status;
    }
    if (inst_size < 2) {
        TVM_RT_SET_ERROR_RETURN(-1, "Instruction size must greater than 1");
    }
    // read instruction list
    uint64_t inst_hash;
    uint64_t inst_op_code;
    if (unlikely(status = reader->ReadBytes(reader, &inst_hash, sizeof(uint64_t)))) {
        return status;
    }
    if (unlikely(status = reader->ReadBytes(reader, &inst_op_code, sizeof(uint64_t)))) {
        return status;
    }

    const size_t num_fields = (size_t)(inst_size - 2);
    const uint64_t *fields =
        (const uint64_t *)reader->ReadToBuffer(reader, sizeof(uint64_t) * num_fields);

    // validate hash
    uint64_t hash_validate = inst_op_code;
    for (size_t i = 0; i < num_fields; ++i) {
        hash_validate = hash__(hash_validate, fields[i]);
    }
    if (unlikely(hash_validate != inst_hash)) {
        TVM_RT_SET_ERROR_RETURN(
            -1, "Instruction hash validate fail, expect %" PRIu64 " but got %" PRIu64, inst_hash,
            hash_validate);
    }
    return TVM_RT_WASM_RelayInstructionCreateFromFields(inst_ptr, inst_op_code,
                                                        (const int64_t *)fields, num_fields);
}

int TVM_RT_WASM_RelayInstructionFree(TVM_RT_WASM_RelayInstruction *inst) {
    switch (inst->op) {
    case RelayOp_Invoke:
        if (inst->op_invoke.reg_invoke_args) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_invoke.reg_invoke_args);
        }
        break;
    case RelayOp_InvokeClosure:
        if (inst->op_invoke_closure.reg_closure_args) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_invoke_closure.reg_closure_args);
        }
        break;
    case RelayOp_InvokePacked:
        if (inst->op_invoke_packed.reg_packed_args) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_invoke_packed.reg_packed_args);
        }
        break;
    case RelayOp_AllocTensor:
        if (inst->op_alloc_tensor.shape) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_alloc_tensor.shape);
        }
        break;
    case RelayOp_AllocADT:
        if (inst->op_alloc_adt.reg_datatype_fields) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_alloc_adt.reg_datatype_fields);
        }
        break;
    case RelayOp_AllocClosure:
        if (inst->op_alloc_closure.reg_free_vars) {
            TVM_RT_WASM_HeapMemoryFree(inst->op_alloc_closure.reg_free_vars);
        }
        break;
    default:
        break;
    }
    return 0;
}