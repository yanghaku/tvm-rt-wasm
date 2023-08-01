/**
 * @file relax_vm/relax_vm_register.h
 * @brief The relax vm register data structure.
 */

#ifndef TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_REGISTER_H_INCLUDE_
#define TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_REGISTER_H_INCLUDE_

#include <tvm/runtime/c_runtime_api.h>

// Special register names
#define RelaxVM_RegName_Special ((RelaxVMRegisterName)(UINT32_C(1) << 31))
#define RelaxVM_RegName_Void ((RelaxVMRegisterName)(RelaxVM_RegName_Special + UINT32_C(0)))
#define RelaxVM_RegName_VM ((RelaxVMRegisterName)(RelaxVM_RegName_Special + UINT32_C(1)))

/**
 * @brief The Relax VM Register Object with reference number.
 */
typedef struct RelaxVMRegisterObject {
    union {
        struct {
            int64_t *shape;
            int ndim;
        } shape_tuple;
        struct {
            void *data;
            DLDevice device;
        } storage;
        struct {
            char *ptr;
            size_t size;
        } string;
    };

    /** @brief The reference number. */
    int ref_num;
} RelaxVMRegisterObject;

/**
 * @brief The Relax VM DLTensor with reference number.
 */
typedef struct RelaxVMRegisterManagedDLTensor {
    /** @brief The DLTensor. */
    DLTensor dl_tensor;
    /** @brief The shape tuple object (if null, check should_free_shape) */
    RelaxVMRegisterObject *shape_obj;
    /** @brief The storage object. (if null, check should_free_storage) */
    RelaxVMRegisterObject *storage_obj;
    /** @brief The reference number. */
    int ref_num;
    /** @brief if the tensor has owned shape or storage */
    bool should_free_shape;
    bool should_free_storage;
} RelaxVMRegisterManagedDLTensor;

/** @brief The relax VM Register value type code. */
typedef enum RelaxVMRegisterTypeCode {
    RelaxVMRegType_Int = kTVMArgInt,
    RelaxVMRegType_Float = kTVMArgFloat,
    RelaxVMRegType_OpaqueHandle = kTVMOpaqueHandle,
    RelaxVMRegType_Nullptr = kTVMNullptr,
    RelaxVMRegType_DataType = kTVMDataType,
    RelaxVMRegType_DLDevice = kDLDevice,
    RelaxVMRegType_DLTensorHandle = kTVMDLTensorHandle,
    RelaxVMRegType_ObjectHandle = kTVMObjectHandle,
    RelaxVMRegType_ModuleHandle = kTVMModuleHandle,
    RelaxVMRegType_PackedFunctionHandle = kTVMPackedFuncHandle,
    RelaxVMRegType_Str = kTVMStr,
    RelaxVMRegType_Bytes = kTVMBytes,

    RelaxVMRegType_ManagedDLTensor = kTVMNDArrayHandle,

// The mask to check if the typecode is Relax VM Object
#define RelaxVMRegType_VMObjectMask (1 << 9)
    // The VM Object types
    RelaxVMRegType_VMObjectStorage = 1 | RelaxVMRegType_VMObjectMask,
    RelaxVMRegType_VMObjectShapeTuple = 2 | RelaxVMRegType_VMObjectMask,
    RelaxVMRegType_VMObjectString = 3 | RelaxVMRegType_VMObjectMask,
} RelaxVMRegisterTypeCode;

/** @brief The relax VM Register to save values. */
typedef struct RelaxVMRegister {
    TVMValue value;
    RelaxVMRegisterTypeCode typecode;
} RelaxVMRegister;

/**
 * @brief Copy the Relax VM Register.
 * @note The dst register must be empty.
 */
#define TVM_RT_WASM_RelaxVMRegisterCopy(_dst, _src)                                                \
    do {                                                                                           \
        (_dst) = (_src);                                                                           \
        if ((_dst).typecode & RelaxVMRegType_VMObjectMask) {                                       \
            ++(((RelaxVMRegisterObject *)((_dst).value.v_handle))->ref_num);                       \
        }                                                                                          \
    } while (0)

/**
 * @brief Create a new Relax VM Object.
 */
#define TVM_RT_WASM_RelaxVMRegisterCreateObject(_dst)                                              \
    do {                                                                                           \
        (_dst) = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegisterObject));                       \
        (_dst)->ref_num = 1;                                                                       \
    } while (0)

/**
 * @brief Create a new Relax Managed DLTensor.
 */
#define TVM_RT_WASM_RelaxVMRegisterCreateManagedDLTensor(_dst)                                     \
    do {                                                                                           \
        (_dst) = TVM_RT_WASM_HeapMemoryAlloc(sizeof(RelaxVMRegisterManagedDLTensor));              \
        (_dst)->ref_num = 1;                                                                       \
    } while (0)

/**
 * @brief Free the Relax VM Object instance.
 */
#define TVM_RT_WASM_RelaxVMRegisterFreeObject(_obj, _typecode)                                     \
    do {                                                                                           \
        if ((--(_obj)->ref_num) == 0) {                                                            \
            switch (_typecode) {                                                                   \
            case RelaxVMRegType_VMObjectStorage:                                                   \
                /* free the Storage */                                                             \
                TVMDeviceFreeDataSpace((_obj)->storage.device, (_obj)->storage.data);              \
                break;                                                                             \
            case RelaxVMRegType_VMObjectShapeTuple:                                                \
                /* free the Shape Tuple */                                                         \
                TVM_RT_WASM_HeapMemoryFree((_obj)->shape_tuple.shape);                             \
                break;                                                                             \
            case RelaxVMRegType_VMObjectString:                                                    \
                /* free the String */                                                              \
                TVM_RT_WASM_HeapMemoryFree((_obj)->string.ptr);                                    \
                break;                                                                             \
            default:                                                                               \
                break;                                                                             \
            }                                                                                      \
            TVM_RT_WASM_HeapMemoryFree((_obj));                                                    \
        }                                                                                          \
    } while (0)

/**
 * @brief Free the Relax VM Register Managed DLTensor.
 */
#define TVM_RT_WASM_RelaxVMRegisterFreeManagedDLTensor(_tensor)                                    \
    do {                                                                                           \
        if ((--(_tensor)->ref_num) == 0) {                                                         \
            if ((_tensor)->shape_obj) {                                                            \
                TVM_RT_WASM_RelaxVMRegisterFreeObject((_tensor)->shape_obj,                        \
                                                      RelaxVMRegType_VMObjectShapeTuple);          \
            } else if ((_tensor)->should_free_shape) {                                             \
                TVM_RT_WASM_HeapMemoryFree((_tensor)->dl_tensor.shape);                            \
            }                                                                                      \
            if ((_tensor)->storage_obj) {                                                          \
                TVM_RT_WASM_RelaxVMRegisterFreeObject((_tensor)->storage_obj,                      \
                                                      RelaxVMRegType_VMObjectStorage);             \
            } else if ((_tensor)->should_free_storage) {                                           \
                TVMDeviceFreeDataSpace((_tensor)->dl_tensor.device, (_tensor)->dl_tensor.data);    \
            }                                                                                      \
            TVM_RT_WASM_HeapMemoryFree((_tensor));                                                 \
        }                                                                                          \
    } while (0)

/** @brief Free the register value. */
#define TVM_RT_WASM_RelaxVMRegisterFreeValue(_reg)                                                 \
    do {                                                                                           \
        if ((_reg).typecode == RelaxVMRegType_ManagedDLTensor) {                                   \
            /* free the DLTensor */                                                                \
            RelaxVMRegisterManagedDLTensor *managed_tensor = (_reg).value.v_handle;                \
            TVM_RT_WASM_RelaxVMRegisterFreeManagedDLTensor(managed_tensor);                        \
            /* Set the reg to NULL. */                                                             \
            (_reg).typecode = RelaxVMRegType_Nullptr;                                              \
        } else if ((_reg).typecode & RelaxVMRegType_VMObjectMask) {                                \
            RelaxVMRegisterObject *vm_obj = (_reg).value.v_handle;                                 \
            /* Free the vm register object. */                                                     \
            TVM_RT_WASM_RelaxVMRegisterFreeObject(vm_obj, (_reg).typecode);                        \
            /* Set the reg to NULL. */                                                             \
            (_reg).typecode = RelaxVMRegType_Nullptr;                                              \
        }                                                                                          \
    } while (0)

#endif // TVM_RT_WASM_BACKENDS_RELAX_VM_RELAX_VM_REGISTER_H_INCLUDE_
