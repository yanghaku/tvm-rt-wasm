/**
 * @file relay_vm/relay_vm_interpreter.c
 * @brief Run relay VM functions.
 */

#include <device/cpu_memory.h>
#include <module/module.h>
#include <relay_vm/relay_instruction.h>
#include <relay_vm/relay_vm.h>
#include <utils/common.h>

#ifndef DEFAULT_FRAME_CAPACITY
#define DEFAULT_FRAME_CAPACITY 16
#endif // !DEFAULT_FRAME_CAPACITY

INLINE int TVM_RT_WASM_RelayVMLoadTensorFirstElem(TVM_RT_WASM_RelayVMRegister *reg, size_t *size) {
    if (unlikely(reg->tp != Reg_BorrowedTensor && reg->tp != Reg_OwnedTensor)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Tensor type mismatch");
    }
    switch (reg->tensor.dtype.bits) {
    case 8:
        *size = (size_t) * (const uint8_t *)reg->tensor.data;
        break;
    case 16:
        *size = (size_t) * (const uint16_t *)reg->tensor.data;
        break;
    case 32:
        *size = (size_t) * (const uint32_t *)reg->tensor.data;
        break;
    case 64:
        *size = (size_t) * (const uint64_t *)reg->tensor.data;
        break;
    default:
        TVM_RT_SET_ERROR_RETURN(-1, "Unsupported tensor data type");
    }
    return 0;
}

INLINE void TVM_RT_WASM_RelayVMPushFrame(TVM_RT_WASM_RelayVirtualMachine vm,
                                         TVM_RT_WASM_RelayInstruction *code, size_t num_registers,
                                         size_t caller_return_register) {
    // push current frame to stack.
    if (vm->frame_stack_size == vm->frame_stack_capacity) {
        if (vm->frame_stack_capacity == 0) {
            vm->frame_stack_capacity = DEFAULT_FRAME_CAPACITY;
        } else {
            vm->frame_stack_capacity <<= 1;
        }
        // realloc
        struct TVM_RT_WASM_RelayVMFrame_st *frames = TVM_RT_WASM_HeapMemoryAlloc(
            sizeof(struct TVM_RT_WASM_RelayVMFrame_st) * vm->frame_stack_capacity);
        if (vm->frame_stack_size != 0) {
            // copy origin frame
            memcpy(frames, vm->frames,
                   sizeof(struct TVM_RT_WASM_RelayVMFrame_st) * vm->frame_stack_size);
        }
        TVM_RT_WASM_HeapMemoryFree(vm->frames);
        vm->frames = frames;
    }
    // save to stack
    vm->frames[vm->frame_stack_size++] = vm->current_frame;

    // set up the new current frame.
    vm->current_frame.pc = 0;
    vm->current_frame.caller_return_register = caller_return_register;
    vm->current_frame.code = code;
    vm->current_frame.num_registers = num_registers;
    vm->current_frame.registers =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVM_RT_WASM_RelayVMRegister) * num_registers);
    memset(vm->current_frame.registers, 0, sizeof(TVM_RT_WASM_RelayVMRegister) * num_registers);
}

INLINE void TVM_RT_WASM_RelayVMPopFrame(TVM_RT_WASM_RelayVirtualMachine vm) {
    if (vm->current_frame.registers) {
        for (size_t i = 0; i < vm->current_frame.num_registers; ++i) {
            TVM_RT_WASM_RelayVMRegisterFree(vm->current_frame.registers + i);
        }
        TVM_RT_WASM_HeapMemoryFree(vm->current_frame.registers);
    }
    vm->current_frame = vm->frames[--vm->frame_stack_size];
}

#define TVM_RT_WASM_RelayNotImpl(op_name)                                                          \
    TVM_RT_SET_ERROR_AND_GOTO(run_fail, "Relay Opcode `%s` is not impl now.", TOSTRING(op_name))

static int TVM_RT_WASM_RelayVMStart(TVM_RT_WASM_RelayVirtualMachine vm) {
    int status = -1;
    size_t origin_frame_size = vm->frame_stack_size;
    TVM_RT_WASM_RelayExecutable exec = vm->exec;
    TVM_RT_WASM_RelayVMFrame *current_frame = &vm->current_frame;
    while (1) {
        const TVM_RT_WASM_RelayInstruction *current_op = current_frame->code + current_frame->pc;

        switch (current_op->op) {
        case RelayOp_Move: {
            TVM_RT_WASM_RelayVMRegister *src_reg_ptr =
                current_frame->registers + current_op->op_move.reg_from;
            current_frame->registers[current_op->reg_dst] = *src_reg_ptr;
            src_reg_ptr->tp = Reg_Null;

            ++current_frame->pc;
            break;
        }
        case RelayOp_Ret: {
            TVM_RT_WASM_RelayVMRegister *ret_ptr =
                current_frame->registers + current_op->op_ret.reg_result;
            TVM_RT_WASM_RelayVMRegister ret_value = *ret_ptr;
            ret_ptr->tp = Reg_Null;

            size_t caller_reg = current_frame->caller_return_register;
            size_t frame_stack_size = vm->frame_stack_size;
            TVM_RT_WASM_RelayVMPopFrame(vm);
            if (frame_stack_size == origin_frame_size) {
                TVM_RT_WASM_RelayVMRegisterFree(&vm->ret_register);
                vm->ret_register = ret_value;
                return 0;
            } else {
                current_frame->registers[caller_reg] = ret_value;
            }
            break;
        }
        case RelayOp_Invoke:
            TVM_RT_WASM_RelayNotImpl(RelayOp_Invoke);
        case RelayOp_InvokeClosure:
            TVM_RT_WASM_RelayNotImpl(RelayOp_InvokeClosure);
        case RelayOp_InvokePacked: {
            int num_args = 0;
            for (size_t i = 0; i < current_op->op_invoke_packed.arity; ++i) {
                TVM_RT_WASM_RelayVMRegister *r =
                    current_frame->registers + current_op->op_invoke_packed.reg_packed_args[i];
                if (r->tp == Reg_OwnedTensor || r->tp == Reg_BorrowedTensor) {
                    ++num_args;
                } else {
                    // todo: ADT
                }
            }
            if (unlikely(current_op->op_invoke_packed.packed_index >=
                         vm->exec->num_packed_functions)) {
                TVM_RT_SET_ERROR_AND_GOTO(run_fail, "Invalid packed function index");
            }
            TVMValue *args = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(TVMValue) * num_args);
            int *arg_type = TVM_RT_WASM_WorkplaceMemoryAlloc(sizeof(int) * num_args);

            num_args = 0;
            for (size_t i = 0; i < current_op->op_invoke_packed.arity; ++i) {
                TVM_RT_WASM_RelayVMRegister *r =
                    current_frame->registers + current_op->op_invoke_packed.reg_packed_args[i];
                if (r->tp == Reg_OwnedTensor || r->tp == Reg_BorrowedTensor) {
                    args[num_args].v_handle = &r->tensor;
                    arg_type[num_args++] = kTVMDLTensorHandle;
                } else {
                    // todo: ADT
                }
            }

            TVMFunctionHandle func =
                vm->exec->packed_functions[current_op->op_invoke_packed.packed_index];
            PackedFunction *pf = (PackedFunction *)func;
            TVMValue ret_val;
            int ret_type_code;
            status = pf->exec(args, arg_type, num_args, &ret_val, &ret_type_code, pf);

            TVM_RT_WASM_WorkplaceMemoryFree(arg_type);
            TVM_RT_WASM_WorkplaceMemoryFree(args);
            if (unlikely(status)) {
                goto run_fail;
            }
            ++current_frame->pc;
            break;
        }
        case RelayOp_AllocTensor: {
            TVM_RT_WASM_RelayVMRegister *reg_dst = current_frame->registers + current_op->reg_dst;
            TVM_RT_WASM_RelayVMRegister *storage =
                current_frame->registers + current_op->op_alloc_tensor.reg_storage;
            size_t offset;
            status = TVM_RT_WASM_RelayVMLoadTensorFirstElem(
                current_frame->registers + current_op->op_alloc_tensor.reg_offset, &offset);
            if (unlikely(status)) {
                goto run_fail;
            }

            reg_dst->tp = Reg_OwnedTensor;
            reg_dst->tensor.device = storage->storage.device;
            reg_dst->tensor.dtype = storage->storage.dtype;
            reg_dst->tensor.data = storage->storage.data;
            reg_dst->tensor.ndim = (int)current_op->op_alloc_tensor.ndim;
            reg_dst->tensor.shape =
                TVM_RT_WASM_HeapMemoryAlloc(reg_dst->tensor.ndim * sizeof(int64_t));
            memcpy(reg_dst->tensor.shape, current_op->op_alloc_tensor.shape,
                   reg_dst->tensor.ndim * sizeof(int64_t));
            reg_dst->tensor.byte_offset = offset;
            storage->tp = Reg_Null;
            ++current_frame->pc;
            break;
        }
        case RelayOp_AllocTensorReg:
            TVM_RT_WASM_RelayNotImpl(RelayOp_AllocTensorReg);
        case RelayOp_AllocADT:
            TVM_RT_WASM_RelayNotImpl(RelayOp_AllocADT);
        case RelayOp_AllocClosure:
            TVM_RT_WASM_RelayNotImpl(RelayOp_AllocClosure);
        case RelayOp_GetField:
            TVM_RT_WASM_RelayNotImpl(RelayOp_GetField);
        case RelayOp_If:
            TVM_RT_WASM_RelayNotImpl(RelayOp_If);
        case RelayOp_LoadConst: {
            TVM_RT_WASM_RelayVMRegister *dst_ptr = current_frame->registers + current_op->reg_dst;
            dst_ptr->tp = Reg_BorrowedTensor;
            dst_ptr->tensor = exec->constant_tensors[current_op->op_load_const.const_index];

            ++current_frame->pc;
            break;
        }
        case RelayOp_Goto:
            TVM_RT_WASM_RelayNotImpl(RelayOp_Goto);
        case RelayOp_GetTag:
            TVM_RT_WASM_RelayNotImpl(RelayOp_GetTag);
        case RelayOp_LoadConstIndex:
            TVM_RT_WASM_RelayNotImpl(RelayOp_LoadConstIndex);
        case RelayOp_Fatal:
            TVM_RT_SET_ERROR_AND_GOTO(run_fail, "Relay VM Fatal");
        case RelayOp_AllocStorage: {
            TVM_RT_WASM_RelayVMRegister *dst_reg = current_frame->registers + current_op->reg_dst;
            size_t nbytes;
            status = TVM_RT_WASM_RelayVMLoadTensorFirstElem(
                current_frame->registers + current_op->op_alloc_storage.reg_allocation_size,
                &nbytes);
            if (unlikely(status)) {
                goto run_fail;
            }

            DLDevice device = exec->devices[current_op->op_alloc_storage.device_index];
            void *data = NULL;
            DLDataType dtype = current_op->op_alloc_storage.dtype_hint;
            status = TVMDeviceAllocDataSpace(device, nbytes, current_op->op_alloc_storage.alignment,
                                             dtype, &data);
            if (unlikely(status)) {
                goto run_fail;
            }

            dst_reg->tp = Reg_Storage;
            dst_reg->storage.device = device;
            dst_reg->storage.dtype = dtype;
            dst_reg->storage.size = nbytes;
            dst_reg->storage.data = data;
            ++current_frame->pc;
            break;
        }
        case RelayOp_ShapeOf:
            TVM_RT_WASM_RelayNotImpl(RelayOp_ShapeOf);
        case RelayOp_ReshapeTensor: {
            TVM_RT_WASM_RelayVMRegister *src_reg =
                current_frame->registers + current_op->op_reshape_tensor.reg_tensor;
            TVM_RT_WASM_RelayVMRegister *dst_reg = current_frame->registers + current_op->reg_dst;
            *dst_reg = *src_reg;
            src_reg->tp = Reg_Null;
            TVM_RT_WASM_RelayVMRegister *new_shape_reg =
                current_frame->registers + current_op->op_reshape_tensor.reg_new_shape;

            TVM_RT_WASM_HeapMemoryFree(dst_reg->tensor.shape);
            int ndim = (int)new_shape_reg->tensor.shape[0];
            dst_reg->tensor.ndim = ndim;
            dst_reg->tensor.shape = TVM_RT_WASM_HeapMemoryAlloc(ndim * sizeof(int64_t));
            memcpy(dst_reg->tensor.shape, new_shape_reg->tensor.data, sizeof(int64_t) * ndim);

            ++current_frame->pc;
            break;
        }
        case RelayOp_DeviceCopy:
            TVM_RT_WASM_RelayNotImpl(RelayOp_DeviceCopy);
        case RelayOp_KillRegister: {
            TVM_RT_WASM_RelayVMRegister *dst_ptr = current_frame->registers + current_op->reg_dst;
            TVM_RT_WASM_RelayVMRegisterFree(dst_ptr);
            dst_ptr->tp = Reg_Null;

            ++current_frame->pc;
            break;
        }
        default:
            TVM_RT_SET_ERROR_AND_GOTO(run_fail, "Invalid Relay Opcode %u", current_op->op);
        }
    }

run_fail:
    // free the frames.
    while (vm->frame_stack_size == origin_frame_size) {
        TVM_RT_WASM_RelayVMPopFrame(vm);
    }
    return status;
}

int TVM_RT_WASM_RelayVMRunFunction(TVM_RT_WASM_RelayVirtualMachine vm,
                                   TVM_RT_WASM_RelayVMFunc func) {
    TVM_RT_WASM_RelayVMPushFrame(vm, func->exec_func->instructions,
                                 func->exec_func->register_file_size, 0);
    for (size_t i = 0; i < func->exec_func->num_params; ++i) {
        TVM_RT_WASM_RelayVMRegister *dst = vm->current_frame.registers + i;
        *dst = func->inputs[i];
        if (dst->tp == Reg_OwnedTensor) {
            dst->tp = Reg_BorrowedTensor;
        }
    }
    return TVM_RT_WASM_RelayVMStart(vm);
}
