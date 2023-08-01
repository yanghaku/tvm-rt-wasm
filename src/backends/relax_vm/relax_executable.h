/**
 * @file relax_vm/relax_executable.h
 * @brief Definition for relax executable and relax executable module.
 * @sa https://github.com/apache/tvm/blob/unity/include/tvm/runtime/relax_vm/bytecode.h
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_EXECUTABLE_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_EXECUTABLE_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <dlpack/dlpack.h>
#include <module/module.h>
#include <relax_vm/relax_vm_register.h>

/**
 * In the Relax VM, offset and register name are size_t/ssize_t.
 * size_t/ssize_t may be 32bit or 64bit.
 */
typedef size_t RelaxVMRegisterName;
typedef ssize_t RelaxVMIndex;

/** @brief Relax Executable Instruction data. */
typedef struct RelaxInstruction {
    union {
        struct {
            /** @brief The arguments data list of the packed function. */
            struct RelaxInstructionCallArg {
                union {
                    RelaxVMRegisterName arg_register;
                    RelaxVMIndex const_idx;
                    int64_t immediate_val;
                    RelaxVMIndex func_idx;
                };
                /** @brief The argument type */
                enum RelaxInstructionCallArgType {
                    RelaxInstructionCallArgType_Register = 0U,
                    RelaxInstructionCallArgType_Immediate = 1U,
                    RelaxInstructionCallArgType_ConstIdx = 2U,
                    RelaxInstructionCallArgType_FuncIdx = 3U
                } arg_type;
            } *args;

            /** @brief The destination register. */
            RelaxVMRegisterName reg_dst;
            /** @brief The index into the function table. */
            RelaxVMIndex func_id;
            /** @brief The number of arguments to the packed function. */
            RelaxVMIndex num_args;
        } op_call;
        struct {
            /** @brief The return result register. */
            RelaxVMRegisterName reg_result;
        } op_ret;
        struct {
            /** @brief The jump offset. */
            RelaxVMIndex pc_offset;
        } op_goto;
        struct {
            /** @brief The register containing the cond value. */
            RelaxVMRegisterName reg_cond;
            /** @brief The program counter offset for the false branch. */
            RelaxVMIndex false_offset;
        } op_if;
    };

    /** @brief The type of Relax Instructions. */
    enum RelaxInstructionType {
        RelaxInstructionType_Call = 1U,
        RelaxInstructionType_Ret = 2U,
        RelaxInstructionType_Goto = 3U,
        RelaxInstructionType_If = 4U,
    } type;
} RelaxInstruction;

/** @brief Relax Executable Constant data. */
typedef struct RelaxConstant {
    /** @brief The constant value */
    union {
        DLTensor dl_tensor;
        RelaxVMRegisterObject register_obj;
        DLDataType dl_datatype;
        int64_t int_value;
    };

    /** @brief The type of Constant. */
    enum RelaxConstantType {
        RelaxConstantType_DLTensor = 0U,
        RelaxConstantType_DLDataType = 1U,
        RelaxConstantType_ShapeTuple = 2U,
        RelaxConstantType_String = 3U,
        RelaxConstantType_Int = 4U,
    } type;
} RelaxConstant;

/** @brief Relax Executable Function information. */
typedef struct RelaxFunctionInfo {
    union {
        struct {
            /** @brief map<param name, param index> */
            Trie *params_map;
            /** @brief The number of parameters can be set. */
            size_t num_params;
            /** @brief The register file size of the function. */
            size_t register_file_size;
            /** @brief The start instruction index of the function. */
            RelaxVMIndex start_instr;
            /** @brief The end instruction index of the function. */
            RelaxVMIndex end_instr;
        } vm_func;
        struct {
            /** @brief The packed function. */
            PackedFunction *pf;
            /** @brief The packed function name. */
            const char *name_ptr;
            size_t name_size;
        } packed_func;
    };

    /** @brief The type of function. */
    enum RelaxFunctionType {
        /** @brief System level packed function */
        RelaxFuncType_Packed = 0U,
        /** @brief Relax VM function. */
        RelaxFuncType_VMFunc = 1U,
        /** @brief Relax VM TIR function. */
        RelaxFuncType_VMTIRFunc = 2U,
    } type;
} RelaxFunctionInfo;

/** @brief Relax Executable struct */
typedef struct RelaxExecutable {
    /** @brief Relax function information list */
    RelaxFunctionInfo *relax_functions;
    /** @brief Relax VM function map: map<functions name, RelaxFunctionInfo*> */
    Trie *relax_vm_functions_map;
    /** @brief Relax constants */
    RelaxConstant *constants;
    /** @brief Relax instructions */
    RelaxInstruction *instructions;
    /** @brief The size of relax function information list. */
    size_t num_relex_functions;
    /** @brief The number of relax constants */
    size_t num_constants;
    /** @brief The number of relax instructions */
    size_t num_instructions;
    /** @brief The max number of call instruction arguments. Used to pre-alloc TVMValue in VM */
    size_t max_num_call_args;
} RelaxExecutable;

/** @brief Relax Executable Module struct */
typedef struct RelaxExecutableModule {
    MODULE_BASE_MEMBER

    /** @brief The relax executable */
    RelaxExecutable exec;
} RelaxExecutableModule;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_EXECUTABLE_H_INCLUDE_
