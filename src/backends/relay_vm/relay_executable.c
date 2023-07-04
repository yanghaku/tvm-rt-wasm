/*!
 * @file relay_vm/relay_executable.c
 * @brief the implementation for relay executable functions.
 * @author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <string.h>

#include <device/device_api.h>
#include <module/module.h>
#include <relay_vm/relay_executable.h>
#include <utils/common.h>
#include <utils/stream_reader.h>
#include <utils/tensor_helper.h>

/*! @brief Magic number for executable byte code */
#define kTVMVMBytecodeMagic (0xD225DE2F4214151DUL)

#define kImmediateConstTag (0)
#define kLateBoundConstTag (1)

#define RELAY_LOADER_CHECK_ERROR(func)                                                             \
    do {                                                                                           \
        status = (func);                                                                           \
        if (unlikely(status))                                                                      \
            return status;                                                                         \
    } while (0)

static int TVM_RT_WASM_RelayExecutableLoadVirtualDeviceSection(TVM_RT_WASM_RelayExecutable exe,
                                                               StreamReader *reader) {
    int status;
    uint64_t dev_size; // virtual devices size.
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &dev_size, sizeof(uint64_t)));

    exe->num_devices = (size_t)dev_size;
    exe->devices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLDevice) * exe->num_devices);
    RELAY_LOADER_CHECK_ERROR(
        reader->ReadBytes(reader, exe->devices, sizeof(DLDevice) * exe->num_devices));

    uint32_t host_device_index;
    status = reader->ReadBytes(reader, &host_device_index, sizeof(uint32_t));
    exe->host_device_index = host_device_index;
    if (exe->host_device_index >= exe->num_devices) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid host device index: %zu", exe->host_device_index);
    }
    return status;
}

static int TVM_RT_WASM_RelayExecutableLoadGlobalSection(TVM_RT_WASM_RelayExecutable exe,
                                                        StreamReader *reader) {
    int status;
    // read std::vector<std::string>
    uint64_t num_globals;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &num_globals, sizeof(uint64_t)));

    TVM_RT_WASM_TrieCreate(&exe->global_map);
    for (uint64_t index = 0; index < num_globals; ++index) {
        uint64_t name_len;
        RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &name_len, sizeof(uint64_t)));
        const char *name = reader->ReadToBuffer(reader, (size_t)name_len);
        if (unlikely(name == NULL)) {
            return -1;
        }
        TVM_RT_WASM_TrieInsertWithLen(exe->global_map, (const uint8_t *)name, name_len,
                                      (void *)index);
    }
    return 0;
}

static int TVM_RT_WASM_RelayExecutableLoadConstantSection(TVM_RT_WASM_RelayExecutable exe,
                                                          StreamReader *reader) {
    int status;
    uint64_t size_to_read;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &size_to_read, sizeof(uint64_t)));
    size_t size = (size_t)size_to_read;

    exe->num_constant_tensors = size;
    exe->constant_tensors = TVM_RT_WASM_HeapMemoryAlloc(sizeof(DLTensor) * size);
    memset(exe->constant_tensors, 0, sizeof(DLTensor) * size);
    for (size_t i = 0; i < size; ++i) {
        uint32_t tag;
        RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &tag, sizeof(uint32_t)));
        if (tag == kImmediateConstTag) {
            // read DLTensor
            RELAY_LOADER_CHECK_ERROR(
                TVM_RT_WASM_DLTensor_LoadFromReader(exe->constant_tensors + i, reader));
        } else if (tag == kLateBoundConstTag) {
            if (!exe->late_bound_constant_names) {
                exe->late_bound_constant_names = TVM_RT_WASM_HeapMemoryAlloc(sizeof(char *) * size);
                memset(exe->late_bound_constant_names, 0, sizeof(char *) * size);
            }
            // read std::string
            uint64_t str_len;
            RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &str_len, sizeof(uint64_t)));
            char *name = TVM_RT_WASM_HeapMemoryAlloc(str_len + 1);
            status = reader->ReadBytes(reader, name, str_len);
            if (unlikely(status)) {
                TVM_RT_WASM_HeapMemoryFree(name);
                return status;
            }
            name[str_len] = 0;
            exe->late_bound_constant_names[i] = name;
        } else {
            TVM_RT_SET_ERROR_RETURN(-1, "Unsupported constant tag: %u", tag);
        }
    }

    uint64_t device_index_size;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &device_index_size, sizeof(uint64_t)));
    if (unlikely(device_index_size != size_to_read)) {
        TVM_RT_SET_ERROR_RETURN(-1, "Constant device index need size %" PRIu64 " but got %" PRIu64,
                                size_to_read, device_index_size);
    }
    exe->constant_device_indices = TVM_RT_WASM_HeapMemoryAlloc(sizeof(size_t) * size);
    const int64_t *ids = (const int64_t *)reader->ReadToBuffer(reader, sizeof(int64_t) * size);
    if (unlikely(ids == NULL)) {
        return -1;
    }
    for (size_t i = 0; i < size; ++i) {
        exe->constant_device_indices[i] = (size_t)ids[i];
    }
    return 0;
}

static int TVM_RT_WASM_RelayExecutableLoadPrimitiveNamesSection(TVM_RT_WASM_RelayExecutable exe,
                                                                StreamReader *reader) {
    int status;
    // read std::vector<std::string>
    uint64_t num_primitive_names;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &num_primitive_names, sizeof(uint64_t)));

    Module *module = (Module *)exe->module_handle;
    size_t num_packed_functions = (size_t)num_primitive_names;
    exe->num_packed_functions = num_packed_functions;
    exe->packed_functions =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVMFunctionHandle) * num_packed_functions);

#ifndef DEFAULT_FUNC_NAME_SIZE
#define DEFAULT_FUNC_NAME_SIZE 256
#endif // !DEFAULT_FUNC_NAME_SIZE
    uint64_t name_buf_size = DEFAULT_FUNC_NAME_SIZE;

    char *name = TVM_RT_WASM_WorkplaceMemoryAlloc(name_buf_size);
    for (size_t index = 0; index < num_packed_functions; ++index) {
        uint64_t name_len;
        if (unlikely(status = reader->ReadBytes(reader, &name_len, sizeof(uint64_t)))) {
            TVM_RT_WASM_WorkplaceMemoryFree(name);
            return status;
        }
        if (unlikely(name_len >= name_buf_size)) { // check name buffer size
            TVM_RT_WASM_WorkplaceMemoryFree(name);
            name_buf_size = name_len + 1;
            name = TVM_RT_WASM_WorkplaceMemoryAlloc(name_buf_size);
        }
        if (unlikely(status = reader->ReadBytes(reader, name, (size_t)name_len))) {
            TVM_RT_WASM_WorkplaceMemoryFree(name);
            return status;
        }
        name[name_len] = 0;
        if (unlikely(status =
                         module->GetFunction(module, name, 1, exe->packed_functions + index))) {
            TVM_RT_WASM_WorkplaceMemoryFree(name);
            return status;
        }
    }
    TVM_RT_WASM_WorkplaceMemoryFree(name);

    // read vector<pair<uint64_t, vector<pair<string, string>>>> attrs
    uint64_t attr_map_size;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &attr_map_size, sizeof(uint64_t)));
    while (attr_map_size--) {
        // attr map key
        RELAY_LOADER_CHECK_ERROR(reader->SkipBytes(reader, sizeof(uint64_t)));
        // atty map value
        uint64_t map_size;
        RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &map_size, sizeof(uint64_t)));
        while (map_size--) {
            uint64_t key_str_size;
            RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &key_str_size, sizeof(uint64_t)));
            RELAY_LOADER_CHECK_ERROR(reader->SkipBytes(reader, key_str_size));
            uint64_t value_str_size;
            RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &value_str_size, sizeof(uint64_t)));
            RELAY_LOADER_CHECK_ERROR(reader->SkipBytes(reader, value_str_size));
        }
    }
    return 0;
}

static int TVM_RT_WASM_RelayExecutableLoadCodeSection(TVM_RT_WASM_RelayExecutable exe,
                                                      StreamReader *reader) {
    int status;
    uint64_t size_to_read;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &size_to_read, sizeof(uint64_t)));
    size_t size = (size_t)size_to_read;

    exe->num_functions = size;
    exe->functions = TVM_RT_WASM_HeapMemoryAlloc(sizeof(TVM_RT_WASM_RelayFunction) * size);
    memset(exe->functions, 0, sizeof(TVM_RT_WASM_RelayFunction) * size);
    while (size--) {
        TVM_RT_WASM_RelayFunction func = NULL;
        // load function and its instructions
        RELAY_LOADER_CHECK_ERROR(TVM_RT_WASM_RelayFunctionCreateFromReader(reader, &func));
        // query function index
        uintptr_t data_to_query;
        status = TVM_RT_WASM_TrieQuery(exe->global_map, (const uint8_t *)func->name,
                                       (void **)&data_to_query);
        if (unlikely(status != TRIE_SUCCESS)) {
            // free func.
            TVM_RT_SET_ERROR_RETURN(-1, "Cannot find function name `%s`", func->name);
        }
        size_t index = (size_t)data_to_query;
        exe->functions[index] = func;
    }
    return 0;
}

int TVM_RT_WASM_RelayExecutableCreateFromReader(TVMModuleHandle module_handle, StreamReader *reader,
                                                const DLDevice *devices, uint32_t num_dev,
                                                TVM_RT_WASM_RelayExecutable *exe_ptr) {
    int status;
    if (module_handle == NULL) {
        RELAY_LOADER_CHECK_ERROR(
            TVM_RT_WASM_ModuleFactory(MODULE_SYSTEM_LIB, sizeof(MODULE_SYSTEM_LIB) - 1, ((void *)0),
                                      0, (Module **)&(module_handle)));
    }

    uint64_t header_magic;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &header_magic, sizeof(uint64_t)));
    if (header_magic != kTVMVMBytecodeMagic) {
        TVM_RT_SET_ERROR_RETURN(-1, "Invalid bytecode magic %" PRIu64, header_magic);
    }

    // version (std::string)
    uint64_t version_str_len;
    RELAY_LOADER_CHECK_ERROR(reader->ReadBytes(reader, &version_str_len, sizeof(uint64_t)));
    RELAY_LOADER_CHECK_ERROR(reader->SkipBytes(reader, version_str_len));

    TVM_RT_WASM_RelayExecutable exe =
        TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct TVM_RT_WASM_RelayExecutable_st));
    memset(exe, 0, sizeof(struct TVM_RT_WASM_RelayExecutable_st));
    exe->module_handle = module_handle;

#define LOAD_SECTION_OR_FAIL(section_name)                                                         \
    do {                                                                                           \
        status = TVM_RT_WASM_RelayExecutableLoad##section_name##Section(exe, reader);              \
        if (unlikely(status)) {                                                                    \
            DBG("Reader " TOSTRING(section_name) "Section fail.");                                 \
            goto load_fail;                                                                        \
        }                                                                                          \
    } while (0)

    LOAD_SECTION_OR_FAIL(VirtualDevice);
    LOAD_SECTION_OR_FAIL(Global);
    LOAD_SECTION_OR_FAIL(Constant);
    LOAD_SECTION_OR_FAIL(PrimitiveNames);
    LOAD_SECTION_OR_FAIL(Code);

    // setup devices;
    for (size_t i = 0; i < exe->num_devices; ++i) {
        DLDeviceType tp = exe->devices[i].device_type;
        if (tp == kDLCPU) { // skip the cpu device
            exe->devices[i].device_id = 0;
            continue;
        }
        int has_found = 0;
        for (uint32_t j = 0; j < num_dev; ++j) {
            if (devices[j].device_type == tp) {
                exe->devices[i].device_id = devices[j].device_id;
                has_found = 1;
                break;
            }
        }
        if (unlikely(has_found == 0)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(
                load_fail,
                "Cannot find a physical device for executable (device_type=%d, device_id=%d).", tp,
                exe->devices[i].device_id);
        }
    }

    // todo: support it
    if (unlikely(exe->late_bound_constant_names)) {
        status = -1;
        TVM_RT_SET_ERROR_AND_GOTO(load_fail, "Late bound constant is not supported now.");
    }

    // setup constant tensors.
    for (size_t i = 0; i < exe->num_constant_tensors; ++i) {
        DLTensor *t = exe->constant_tensors + i;
        if (unlikely(t == NULL)) {
            status = -1;
            TVM_RT_SET_ERROR_AND_GOTO(load_fail, "Constant tensors %zu is not loaded.", i);
        }
        DLDevice dev = exe->devices[exe->constant_device_indices[i]];
        if (dev.device_type != kDLCPU) {
            DLTensor origin_tensor = *t;
            t->device = dev;
            DeviceAPI *device_api;
            if (unlikely(status = TVM_RT_WASM_DeviceAPIGet(dev.device_type, &device_api))) {
                goto load_fail;
            }
            size_t nbytes = TVM_RT_WASM_DLTensor_GetDataBytes(t);
            t->data = device_api->AllocDataSpace(dev.device_id, nbytes, 0, t->dtype);
            if (unlikely(t->data == NULL)) {
                status = -1;
                TVM_RT_WASM_HeapMemoryFree(origin_tensor.data);
                goto load_fail;
            }
            if (unlikely(status = device_api->CopyDataFromTo(&origin_tensor, t, NULL))) {
                TVM_RT_WASM_HeapMemoryFree(origin_tensor.data);
                goto load_fail;
            }
            TVM_RT_WASM_HeapMemoryFree(origin_tensor.data);
        }
    }

    // success
    *exe_ptr = exe;
    return 0;

#undef RUN_OR_FAIL
load_fail:
    TVM_RT_WASM_RelayExecutableFree(exe);
    return status;
}

int TVM_RT_WASM_RelayExecutableFree(TVM_RT_WASM_RelayExecutable exe) {
    if (exe->devices) {
        TVM_RT_WASM_HeapMemoryFree(exe->devices);
    }
    if (exe->global_map) {
        TVM_RT_WASM_TrieRelease(exe->global_map);
    }
    if (exe->packed_functions) {
        TVM_RT_WASM_HeapMemoryFree(exe->packed_functions);
    }
    if (exe->constant_tensors) {
        for (size_t i = 0; i < exe->num_constant_tensors; ++i) {
            DLTensor *t = exe->constant_tensors + i;
            if (t->shape) {
                TVM_RT_WASM_HeapMemoryFree(t->shape);
            }
            if (t->data) {
                TVMDeviceFreeDataSpace(t->device, t->data);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(exe->constant_tensors);
    }
    if (exe->late_bound_constant_names) {
        for (size_t i = 0; i < exe->num_constant_tensors; ++i) {
            char *p = exe->late_bound_constant_names[i];
            if (p) {
                TVM_RT_WASM_HeapMemoryFree(p);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(exe->late_bound_constant_names);
    }
    if (exe->constant_device_indices) {
        TVM_RT_WASM_HeapMemoryFree(exe->constant_device_indices);
    }
    if (exe->functions) {
        for (size_t i = 0; i < exe->num_functions; ++i) {
            if (exe->functions[i]) {
                TVM_RT_WASM_RelayFunctionFree(exe->functions[i]);
            }
        }
        TVM_RT_WASM_HeapMemoryFree(exe->functions);
    }
    TVM_RT_WASM_HeapMemoryFree(exe);
    return 0;
}
