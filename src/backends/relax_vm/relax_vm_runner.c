/**
 * @file relax_vm/relax_vm_runner.c
 * @brief Run the relax vm functions.
 */

#include <relax_vm/relax_vm.h>
#include <utils/common.h>

#define TVM_RT_WASM_RelaxVMDefaultFrameCapacity 8

/**
 * Create a new frame and push to VM frame stack, return the pointer to new frame.
 * @param vm The Relax VM instance.
 * @param num_registers The number of registers in the new frame.
 * @return The pointer to created frame.
 */
INLINE RelaxVMFrame *TVM_RT_WASM_RelaxVMFramePush(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                  size_t num_registers) {
    if (unlikely(vm->frame_size == vm->frame_capacity)) {
        if (vm->frame_capacity == 0) {
            // alloc
            vm->frame_capacity = TVM_RT_WASM_RelaxVMDefaultFrameCapacity;
            vm->frames = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMFrame) * vm->frame_capacity);
            memset(vm->frames, 0, sizeof(RelaxVMFrame) * vm->frame_capacity);
        } else {
            // realloc and copy
            vm->frame_capacity <<= 1;
            RelaxVMFrame *frames =
                TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMFrame) * vm->frame_capacity);
            // vm.frame_size * 2 == vm.frame_capacity.
            memcpy(frames, vm->frames, sizeof(RelaxVMFrame) * vm->frame_size);
            memset(frames + vm->frame_size, 0, sizeof(RelaxVMFrame) * vm->frame_size);
            TVM_RT_WASM_HeapMemoryFree(vm->frames);
            vm->frames = frames;
        }
    }
    RelaxVMFrame *frame = vm->frames + (vm->frame_size++);
    if (frame->register_capacity < num_registers) { // find a buffer or alloc.
        for (size_t i = vm->frame_size; i < vm->frame_capacity; ++i) {
            // vm->frames[i].register_size is 0
            if (vm->frames[i].register_capacity >= num_registers) {
                // swap
                RelaxVMRegister *_tmp_reg = vm->frames[i].registers;
                vm->frames[i].registers = frame->registers;
                frame->registers = _tmp_reg;
                size_t _tmp_cap = vm->frames[i].register_capacity;
                vm->frames[i].register_capacity = frame->register_capacity;
                frame->register_capacity = _tmp_cap;
                break;
            }
        }
        if (frame->register_capacity < num_registers) {
            if (frame->registers) {
                TVM_RT_WASM_HeapMemoryFree(frame->registers);
            }
            frame->registers = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegister) * num_registers);
            memset(frame->registers, 0, sizeof(RelaxVMRegister) * num_registers);
            frame->register_capacity = num_registers;
        }
    }
    frame->register_size = num_registers;
    frame->return_pc = vm->pc;
    return frame;
}

/**
 * Pop the frame from VM frame stack.
 * Just free all register values, Do not free register buffers.
 * @param vm The Relax VM instance.
 */
#define TVM_RT_WASM_RelaxVMFramePop(_vm)                                                           \
    do {                                                                                           \
        RelaxVMFrame *_frame = (_vm)->frames + (--((_vm)->frame_size));                            \
        (_vm)->pc = _frame->return_pc;                                                             \
        for (size_t _reg_i = 0; _reg_i < _frame->register_size; ++_reg_i) {                        \
            TVM_RT_WASM_RelaxVMRegisterFreeValue(_frame->registers[_reg_i]);                       \
        }                                                                                          \
        _frame->register_size = 0;                                                                 \
    } while (0)

/**
 * @brief Copy the register from the frame.
 * @note The dst register must be a empty register.
 */
#define TVM_RT_WASM_RelaxRegisterCopyFromFrame(_dst, _src_reg_array, _src_reg_name)                \
    do {                                                                                           \
        if ((_src_reg_name) < RelaxVM_RegName_Special) {                                           \
            TVM_RT_WASM_RelaxVMRegisterCopy(_dst, (_src_reg_array)[_src_reg_name]);                \
        } else if ((_src_reg_name) == RelaxVM_RegName_Void) {                                      \
            (_dst).typecode = RelaxVMRegType_Nullptr;                                              \
        } else if ((_src_reg_name) == RelaxVM_RegName_VM) {                                        \
            (_dst).value.v_handle = vm;                                                            \
            (_dst).typecode = RelaxVMRegType_ObjectHandle;                                         \
        } else {                                                                                   \
            unreachable();                                                                         \
        }                                                                                          \
    } while (0)

static int TVM_RT_WASM_RelaxVMInterpretInstructions(TVM_RT_WASM_RelaxVirtualMachine vm,
                                                    RelaxVMRegister *return_register) {
    RelaxVMFrame *current_frame = vm->frames + (vm->frame_size - 1);
    RelaxVMRegister *registers = current_frame->registers;
    while (1) {
        const RelaxInstruction *instr = &vm->exec_module->exec.instructions[vm->pc];
        switch (instr->type) {
        case RelaxInstructionType_Call: {
            ++vm->pc;
            RelaxFunctionInfo *func =
                vm->exec_module->exec.relax_functions + instr->op_call.func_id;
            switch (func->type) {
            case RelaxFuncType_Packed: {
                int num_args = (int)instr->op_call.num_args;
                for (int i = 0; i < num_args; ++i) {
                    const struct RelaxInstructionCallArg *arg = instr->op_call.args + i;
                    switch (arg->arg_type) {
                    case RelaxInstructionCallArgType_ConstIdx: {
                        vm->call_packed_args_value[i] = vm->constants[arg->const_idx].value;
                        vm->call_packed_args_typecode[i] = vm->constants[arg->const_idx].typecode;
                        break;
                    }
                    case RelaxInstructionCallArgType_Immediate:
                        vm->call_packed_args_value[i].v_int64 = arg->immediate_val;
                        vm->call_packed_args_typecode[i] = kTVMArgInt;
                        break;
                    case RelaxInstructionCallArgType_Register: {
                        RelaxVMRegisterName reg_name = arg->arg_register;
                        if (reg_name < RelaxVM_RegName_Special) {
                            vm->call_packed_args_value[i] = registers[reg_name].value;
                            vm->call_packed_args_typecode[i] = registers[reg_name].typecode;
                        } else if (reg_name == RelaxVM_RegName_Void) {
                            vm->call_packed_args_value[i].v_handle = NULL;
                            vm->call_packed_args_typecode[i] = kTVMNullptr;
                        } else if (reg_name == RelaxVM_RegName_VM) {
                            vm->call_packed_args_value[i].v_handle = vm;
                            vm->call_packed_args_typecode[i] = kTVMObjectHandle;
                        } else {
                            unreachable();
                        }
                        break;
                    }
                    case RelaxInstructionCallArgType_FuncIdx:
                    default:
                        unreachable();
                    }
                }
                if (func->packed_func.pf) {
                    TVMValue ret_value;
                    RelaxVMRegisterTypeCode ret_code;
                    int status = func->packed_func.pf->exec(vm->call_packed_args_value,
                                                            vm->call_packed_args_typecode, num_args,
                                                            &ret_value, (int *)&ret_code, NULL);

                    if (unlikely(status)) {
                        return status;
                    }
                    if (instr->op_call.reg_dst < current_frame->register_size) {
                        RelaxVMRegister *reg = registers + instr->op_call.reg_dst;
                        TVM_RT_WASM_RelaxVMRegisterFreeValue(*reg);
                        reg->typecode = ret_code;
                        reg->value = ret_value;
                    }
                } else { // The null value function, clear the dst register.
                    RelaxVMRegister *reg = registers + instr->op_call.reg_dst;
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(*reg);
                }
                break;
            }
            case RelaxFuncType_VMFunc: {
                TVM_RT_WASM_RelaxVMFramePush(vm, func->vm_func.register_file_size);
                vm->pc = func->vm_func.start_instr;
                RelaxVMFrame *callee_frame = vm->frames + (vm->frame_size - 1);
                RelaxVMRegister *callee_registers = callee_frame->registers;
                callee_frame->reg_caller_return = instr->op_call.reg_dst;
                for (size_t i = 0; i < func->vm_func.num_params; ++i) {
                    const struct RelaxInstructionCallArg *arg = instr->op_call.args + i;
                    switch (arg->arg_type) {
                    case RelaxInstructionCallArgType_ConstIdx:
                        TVM_RT_WASM_RelaxVMRegisterCopy(callee_registers[i],
                                                        vm->constants[arg->const_idx]);
                        break;
                    case RelaxInstructionCallArgType_Immediate:
                        callee_registers[i].typecode = RelaxVMRegType_Int;
                        callee_registers[i].value.v_int64 = arg->immediate_val;
                        break;
                    case RelaxInstructionCallArgType_Register: {
                        TVM_RT_WASM_RelaxRegisterCopyFromFrame(callee_registers[i], registers,
                                                               arg->arg_register);
                        break;
                    }
                    case RelaxInstructionCallArgType_FuncIdx:
                    default:
                        unreachable();
                    }
                }
                registers = callee_registers;
                current_frame = callee_frame;
                break;
            }
            case RelaxFuncType_VMTIRFunc:
            default:
                unreachable();
            }
            break;
        }
        case RelaxInstructionType_Ret: {
            if (vm->frame_size <= 1) {
                // free the ret register old value.
                TVM_RT_WASM_RelaxVMRegisterFreeValue(*return_register);
                TVM_RT_WASM_RelaxRegisterCopyFromFrame(*return_register, registers,
                                                       instr->op_ret.reg_result);
                TVM_RT_WASM_RelaxVMFramePop(vm);
                return 0;
            } else {
                RelaxVMFrame *caller_frame = vm->frames + (vm->frame_size - 2);
                RelaxVMRegister *caller_registers = caller_frame->registers;
                RelaxVMRegisterName reg_rt = current_frame->reg_caller_return;
                if (likely(reg_rt < caller_frame->register_size)) {
                    // free the ret register old value.
                    TVM_RT_WASM_RelaxVMRegisterFreeValue(caller_registers[reg_rt]);
                    // copy to ret register
                    TVM_RT_WASM_RelaxRegisterCopyFromFrame(caller_registers[reg_rt], registers,
                                                           instr->op_ret.reg_result);
                }
                TVM_RT_WASM_RelaxVMFramePop(vm);
                current_frame = caller_frame;
                registers = caller_registers;
            }
            break;
        }
        case RelaxInstructionType_Goto:
            vm->pc += instr->op_goto.pc_offset;
            break;
        case RelaxInstructionType_If: {
            // assert(instr->op_if.reg_cond < RelaxVM_RegName_Special);
            RelaxVMRegister cond = registers[instr->op_if.reg_cond];
            // assert(cond.typecode == RelaxVMRegType_Int);
            if (cond.value.v_int64 != 0) {
                ++vm->pc;
            } else {
                vm->pc += instr->op_if.false_offset;
            }
            break;
        }
        default:
            unreachable();
        }
    }
}

int TVM_RT_WASM_RelaxVMRunFunction(TVM_RT_WASM_RelaxVirtualMachine vm, RelaxFunctionInfo *func,
                                   RelaxVMFunctionInputsOutput *inputs_output) {
    RelaxVMFrame *current_frame =
        TVM_RT_WASM_RelaxVMFramePush(vm, func->vm_func.register_file_size);
    vm->pc = func->vm_func.start_instr;
    // set input
    for (size_t i = 0; i < inputs_output->num_inputs; ++i) {
        TVM_RT_WASM_RelaxVMRegisterCopy(current_frame->registers[i],
                                        inputs_output->inputs_output[i]);
    }
    return TVM_RT_WASM_RelaxVMInterpretInstructions(
        vm, &inputs_output->inputs_output[inputs_output->num_inputs]);
}
