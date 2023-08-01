/**
 * @file relax_vm/relax_executable_module.c
 * @brief The implementation for relax executable module.
 */

#include <string.h>

#include <module/module_impl.h>
#include <relax_vm/relax_executable.h>
#include <utils/stream_reader.h>
#include <utils/tensor_helper.h>

/** @brief Magic number for executable byte code */
#define kTVMVMBytecodeMagic (UINT64_C(0xD225DE2F4214151D))

/**
 * TVM special register name.
 * In this implementation, size_t may 32bit, so we must change the special register < 32bit.
 */
#define kTVM_kBeginSpecialReg ((int64_t)(INT64_C(1) << 54))
#define kTVM_kVoidRegister ((int64_t)(kTVM_kBeginSpecialReg + INT64_C(0)))
#define kTVM_kVMRegister ((int64_t)(kTVM_kBeginSpecialReg + INT64_C(1)))

// Definitions for relax call instruction argument data.
#define RelaxInstructionCallArg_TypeEnumBits 8
#define RelaxInstructionCallArg_ValueBits                                                          \
    (sizeof(int64_t) * 8 - RelaxInstructionCallArg_TypeEnumBits)
#define RelaxInstructionCallArg_ValueMask ((INT64_C(1) << RelaxInstructionCallArg_ValueBits) - 1)
// Split data to type and value.
#define RelaxInstructionCallArgDataGetType(_data)                                                  \
    (enum RelaxInstructionCallArgType)(((_data) >> RelaxInstructionCallArg_ValueBits) & 0xFF)
#define RelaxInstructionCallArgDataGetValue(_data)                                                 \
    ((((_data)&RelaxInstructionCallArg_ValueMask) << RelaxInstructionCallArg_TypeEnumBits) >>      \
     RelaxInstructionCallArg_TypeEnumBits)

static int TVM_RT_WASM_RelaxExecutableLoadGlobalSection(RelaxExecutable *exec,
                                                        BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
    size_t func_size = (size_t) * (uint64_t *)cur_ptr;
    TVM_RT_WASM_TrieCreate(&exec->relax_vm_functions_map);
    exec->relax_functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxFunctionInfo) * func_size);
    exec->num_relex_functions = func_size;
    memset(exec->relax_functions, 0, sizeof(RelaxFunctionInfo) * func_size);
    for (size_t index = 0; index < func_size; ++index) {
        RelaxFunctionInfo *info = exec->relax_functions + index;
        // kind
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint32_t), load_global_fail);
        info->type = (enum RelaxFunctionType) * (uint32_t *)cur_ptr;

        // name
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
        size_t name_size = (size_t) * (uint64_t *)cur_ptr;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, load_global_fail);
        const char *name = cur_ptr;

        switch (info->type) {
        case RelaxFuncType_Packed: {
#define READ_AND_CHECK_EQ(_msg, _expected)                                                         \
    do {                                                                                           \
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_global_fail);             \
        const int64_t val = *(const int64_t *)cur_ptr;                                             \
        if (unlikely(val != (_expected))) {                                                        \
            status = -1;                                                                           \
            TVM_RT_SET_ERROR_AND_GOTO(load_global_fail,                                            \
                                      "Expect relax function.%s %" PRIi64 " but got %" PRIi64,     \
                                      (_msg), (_expected), val);                                   \
        }                                                                                          \
    } while (0)

            READ_AND_CHECK_EQ("start_instr", INT64_C(0));
            READ_AND_CHECK_EQ("end_instr", INT64_C(0));
            READ_AND_CHECK_EQ("num_args", INT64_C(-2));
            READ_AND_CHECK_EQ("register_file_size", INT64_C(0));
            READ_AND_CHECK_EQ("num_params", INT64_C(0));
            info->packed_func.name_ptr = name;
            info->packed_func.name_size = name_size;
            break;
        }
        case RelaxFuncType_VMFunc:
            // insert relax VM function to relax function maps
            TVM_RT_WASM_TrieInsertWithLen(exec->relax_vm_functions_map, (const uint8_t *)name,
                                          name_size, (void *)info);
            // start and end instruction index
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_global_fail);
            info->vm_func.start_instr = (RelaxVMIndex) * (int64_t *)cur_ptr;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_global_fail);
            info->vm_func.end_instr = (RelaxVMIndex) * (int64_t *)cur_ptr;
            // number of arguments
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_global_fail);
            const int64_t num_args = *(int64_t *)cur_ptr;
            // register file size
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
            uint64_t register_size = *(uint64_t *)cur_ptr;
            if (unlikely(register_size >= (uint64_t)RelaxVM_RegName_Special)) {
                TVM_RT_SET_ERROR_AND_GOTO(load_global_fail,
                                          "Relax VM Register Name is too big: %" PRIu64 " >= %zu",
                                          register_size, RelaxVM_RegName_Special);
            }
            info->vm_func.register_file_size = (size_t)register_size;
            // param names
            READ_AND_CHECK_EQ("num_params", num_args);

            TVM_RT_WASM_TrieCreate(&info->vm_func.params_map);
            info->vm_func.num_params = (size_t)num_args;
            for (size_t param_i = 0; param_i < info->vm_func.num_params; ++param_i) {
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_global_fail);
                name_size = (size_t) * (uint64_t *)cur_ptr;
                TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, name_size, load_global_fail);
                status = TVM_RT_WASM_TrieInsertWithLen(info->vm_func.params_map,
                                                       (const uint8_t *)cur_ptr, name_size,
                                                       (void *)(uintptr_t)param_i);
            }
            break;
#undef READ_AND_CHECK_EQ
        case RelaxFuncType_VMTIRFunc:
            TVM_RT_SET_ERROR_AND_GOTO(load_global_fail, "Unsupported VM TIR function now");
        default:
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_global_fail, "Unsupported relax function type %u.",
                                      info->type);
        }
    }
load_global_fail:
    return status;
}

static int TVM_RT_WASM_RelaxExecutableLoadConstantSection(RelaxExecutable *exec,
                                                          BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_constant_fail);
    const size_t num_constants = (const size_t) * (uint64_t *)cur_ptr;
    exec->num_constants = num_constants;
    exec->constants = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxConstant) * num_constants);
    for (size_t c_id = 0; c_id < num_constants; ++c_id) {
        RelaxConstant *constant = exec->constants + c_id;
        TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint32_t), load_constant_fail);
        constant->type = (enum RelaxConstantType) * (uint32_t *)cur_ptr;
        switch (constant->type) {
        case RelaxConstantType_DLTensor:
            status = TVM_RT_WASM_DLTensor_LoadFromBinary(&constant->dl_tensor, reader);
            if (unlikely(status)) {
                goto load_constant_fail;
            }
            break;
        case RelaxConstantType_DLDataType:
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(DLDataType), load_constant_fail);
            constant->dl_datatype = *(DLDataType *)cur_ptr;
            break;
        case RelaxConstantType_ShapeTuple:
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_constant_fail);
            constant->register_obj.shape_tuple.ndim = (int)*(int64_t *)cur_ptr;
            TVM_RT_WASM_BinaryCheckReadOrGoto(
                cur_ptr, constant->register_obj.shape_tuple.ndim * sizeof(uint64_t),
                load_constant_fail);
            constant->register_obj.shape_tuple.shape = (int64_t *)cur_ptr;
            constant->register_obj.ref_num = 1;
            break;
        case RelaxConstantType_String:
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_constant_fail);
            constant->register_obj.string.size = (size_t) * (int64_t *)cur_ptr;
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, constant->register_obj.string.size,
                                              load_constant_fail);
            constant->register_obj.string.ptr = (char *)cur_ptr;
            constant->register_obj.ref_num = 1;
            break;
        case RelaxConstantType_Int:
            TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_constant_fail);
            constant->int_value = *(int64_t *)cur_ptr;
            break;
        default:
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_constant_fail, "Unsupported relax constant type %u",
                                      constant->type);
        }
    }
load_constant_fail:
    return status;
}

static int TVM_RT_WASM_RelaxExecutableLoadCodeSection(RelaxExecutable *exec, BinaryReader *reader) {
    int status = 0;
    const char *cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_code_fail);
    const size_t instr_offset_size = (size_t) * (int64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * instr_offset_size, load_code_fail);
    const int64_t *instr_offsets = (const int64_t *)cur_ptr;

    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t), load_code_fail);
    const size_t instr_data_size = (size_t) * (int64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(int64_t) * instr_data_size, load_code_fail);
    const int64_t *instr_data = (const int64_t *)cur_ptr;

    exec->num_instructions = instr_offset_size;
    exec->instructions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxInstruction) * exec->num_instructions);
    memset(exec->instructions, 0, sizeof(RelaxInstruction) * exec->num_instructions);
    for (size_t i = 0; i < instr_offset_size; ++i) {
        RelaxInstruction *instr = exec->instructions + i;
        size_t offset = (size_t)instr_offsets[i];

#define CHECK_CONVERT_REG_OR_FAIL(_dst, _val)                                                      \
    do {                                                                                           \
        int64_t _tmp = (_val);                                                                     \
        if (_tmp < kTVM_kBeginSpecialReg) {                                                        \
            if (unlikely(_tmp >= (int64_t)RelaxVM_RegName_Special)) { /* overflow */               \
                TVM_RT_SET_ERROR_RETURN(-1,                                                        \
                                        "Relax VM Register Name is too big: %" PRIi64 " >= %zu",   \
                                        _tmp, RelaxVM_RegName_Special);                            \
            }                                                                                      \
            (_dst) = (RelaxVMRegisterName)_tmp;                                                    \
        } else if (_tmp == kTVM_kVoidRegister) {                                                   \
            (_dst) = RelaxVM_RegName_Void;                                                         \
        } else if (_tmp == kTVM_kVMRegister) {                                                     \
            (_dst) = RelaxVM_RegName_VM;                                                           \
        } else {                                                                                   \
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported special relax VM register %" PRIi64, _tmp);   \
        }                                                                                          \
    } while (0)

        instr->type = (enum RelaxInstructionType)instr_data[offset];
        switch (instr->type) {
        case RelaxInstructionType_Call:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_call.reg_dst, instr_data[offset + 1]);
            instr->op_call.func_id = (RelaxVMIndex)instr_data[offset + 2];
            instr->op_call.num_args = (RelaxVMIndex)instr_data[offset + 3];
            const int64_t *args_data = &instr_data[offset + 4];
            size_t max_arg_offset = offset + 4 + instr->op_call.num_args;
            if (unlikely(max_arg_offset > instr_data_size)) {
                TVM_RT_SET_ERROR_RETURN(-1, "Instruction Call num args exceed: %zu > %zu",
                                        max_arg_offset, instr_data_size);
            }
            // check and parse args
            size_t num_args = instr->op_call.num_args;
            exec->max_num_call_args = MAX(num_args, exec->max_num_call_args);
            struct RelaxInstructionCallArg *args =
                TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct RelaxInstructionCallArg) * num_args);
            instr->op_call.args = args;
            for (size_t arg_i = 0; arg_i < num_args; ++arg_i) {
                int64_t arg_data = args_data[arg_i];
                args[arg_i].arg_type = RelaxInstructionCallArgDataGetType(arg_data);
                int64_t arg_value = RelaxInstructionCallArgDataGetValue(arg_data);

                switch (args[arg_i].arg_type) {
                case RelaxInstructionCallArgType_Register: {
                    CHECK_CONVERT_REG_OR_FAIL(args[arg_i].arg_register, arg_value);
                    break;
                }
                case RelaxInstructionCallArgType_Immediate:
                    args[arg_i].immediate_val = arg_value;
                    break;
                case RelaxInstructionCallArgType_ConstIdx: {
                    size_t const_id = (size_t)arg_value;
                    if (const_id >= exec->num_constants) {
                        TVM_RT_SET_ERROR_RETURN(-1, "Constants index exceed: %zu >= %zu.", const_id,
                                                exec->num_constants);
                    }
                    args[arg_i].const_idx = (RelaxVMIndex)const_id;
                    break;
                }
                case RelaxInstructionCallArgType_FuncIdx: {
                    size_t func_id = (size_t)arg_value;
                    if (func_id >= exec->num_relex_functions) {
                        TVM_RT_SET_ERROR_RETURN(-1, "Functions index exceed: %zu >= %zu.", func_id,
                                                exec->num_relex_functions);
                    }
                    args[arg_i].func_idx = (RelaxVMIndex)func_id;
                    // todo
                    TVM_RT_SET_ERROR_RETURN(-1, "cannot pass functions in args now.");
                }
                default:
                    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported instruction call arg type %u.",
                                            args[arg_i].arg_type);
                }
            }
            break;
        case RelaxInstructionType_Ret:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_ret.reg_result, instr_data[offset + 1]);
            break;
        case RelaxInstructionType_Goto:
            instr->op_goto.pc_offset = (RelaxVMIndex)instr_data[offset + 1];
            break;
        case RelaxInstructionType_If:
            CHECK_CONVERT_REG_OR_FAIL(instr->op_if.reg_cond, instr_data[offset + 1]);
            instr->op_if.false_offset = (RelaxVMIndex)instr_data[offset + 2];
            break;
        default:
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported relax instruction type %u", instr->type);
        }
    }
#undef CHECK_CONVERT_REG_OR_FAIL
load_code_fail:
    return status;
}

static int TVM_RT_WASM_RelaxExecutableModuleReleaseFunc(Module *self) {
    RelaxExecutableModule *mod = (RelaxExecutableModule *)self;
    MODULE_BASE_MEMBER_FREE(mod);

    if (mod->exec.constants) {
        TVM_RT_WASM_HeapMemoryFree(mod->exec.constants);
    }
    if (mod->exec.relax_vm_functions_map) {
        TVM_RT_WASM_TrieRelease(mod->exec.relax_vm_functions_map);
    }
    if (mod->exec.relax_functions) {
        RelaxFunctionInfo *funcs = mod->exec.relax_functions;
        for (size_t i = 0; i < mod->exec.num_relex_functions; ++i) {
            RelaxFunctionInfo *func = funcs + i;
            switch (func->type) {
            case RelaxFuncType_VMFunc:
                if (func->vm_func.params_map) {
                    TVM_RT_WASM_TrieRelease(func->vm_func.params_map);
                }
                break;
            case RelaxFuncType_Packed:
            case RelaxFuncType_VMTIRFunc:
                break;
            default:
                unreachable();
            }
        }
        TVM_RT_WASM_HeapMemoryFree(funcs);
    }
    if (mod->exec.instructions) {
        RelaxInstruction *instructions = mod->exec.instructions;
        for (size_t i = 0; i < mod->exec.num_instructions; ++i) {
            RelaxInstruction *instr = instructions + i;
            if (instr->type == RelaxInstructionType_Call) {
                if (instr->op_call.args) {
                    TVM_RT_WASM_HeapMemoryFree(instr->op_call.args);
                }
            }
        }
        TVM_RT_WASM_HeapMemoryFree(instructions);
    }

    TVM_RT_WASM_HeapMemoryFree(mod);
    return 0;
}

int TVM_RT_WASM_RelaxExecutableModuleCreate(BinaryReader *reader, Module **out) {
    *out = NULL;
    int status;
    const char *cur_ptr;

    // skip the exec module size
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    // check magic and version
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    uint64_t header_magic = *(uint64_t *)cur_ptr;
    if (unlikely(header_magic != kTVMVMBytecodeMagic)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid bytecode magic %" PRIu64, header_magic);
    }
    // version (std::string)
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t), load_fail);
    size_t version_str_len = (size_t) * (uint64_t *)cur_ptr;
    TVM_RT_WASM_BinaryCheckReadOrGoto(cur_ptr, version_str_len, load_fail);

    // Allocate memory for module instance
    RelaxExecutableModule *exec_mod = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxExecutableModule));
    memset(exec_mod, 0, sizeof(RelaxExecutableModule));
    *out = (Module *)exec_mod;

    exec_mod->Release = TVM_RT_WASM_RelaxExecutableModuleReleaseFunc;
    exec_mod->GetFunction = TVM_RT_WASM_DefaultModuleGetFunction;

#define RelaxLoadSection(_section_name)                                                            \
    do {                                                                                           \
        status = TVM_RT_WASM_RelaxExecutableLoad##_section_name##Section(&exec_mod->exec, reader); \
        if (unlikely(status)) {                                                                    \
            DBG("Relax Executable Load " TOSTRING(_section_name) " Section fail.");                \
            return status;                                                                         \
        }                                                                                          \
    } while (0)

    // load relax executable global section
    RelaxLoadSection(Global);
    // load relax executable constant section
    RelaxLoadSection(Constant);
    // load relax executable code section
    RelaxLoadSection(Code);

#undef RelaxLoadSection
    return 0;

load_fail:
    if (*out) {
        TVM_RT_WASM_RelaxExecutableModuleReleaseFunc(*out);
        *out = NULL;
    }
    return status;
}
