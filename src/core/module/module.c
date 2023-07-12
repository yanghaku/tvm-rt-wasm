/**
 * @file module/module.c
 * @brief Implement functions for module_impl.h
 */

#include <string.h>

#include <device/cpu_memory.h>
#include <module/module_impl.h>
#include <utils/binary_reader.h>

#define MODULE_CREATE_IF_NO_SUPPORT(dev)                                                           \
    _Pragma(TOSTRING(weak TVM_RT_WASM_##dev##ModuleCreate));                                       \
    int TVM_RT_WASM_##dev##ModuleCreate(BinaryReader *reader, Module **out) {                      \
        (void)reader;                                                                              \
        *out = NULL;                                                                               \
        TVM_RT_##dev##_NOT_LINK();                                                                 \
        return -1;                                                                                 \
    }

MODULE_CREATE_IF_NO_SUPPORT(CUDA)
MODULE_CREATE_IF_NO_SUPPORT(WebGPU)

#define TVM_RT_BACKEND_NOT_ON(_mod, _backend)                                                      \
    do {                                                                                           \
        fprintf(stderr,                                                                            \
                "%s module is not supported! You can link with the backend library `%s`.\n",       \
                TOSTRING(_mod), _backend);                                                         \
        exit(-1);                                                                                  \
    } while (0)

#define TVM_RT_RelaxExecutable_NOT_LINK()                                                          \
    TVM_RT_BACKEND_NOT_ON(RelaxExecutable, "tvm-rt-backend-relax-vm")

MODULE_CREATE_IF_NO_SUPPORT(RelaxExecutable)

/** @brief Default function for module get function. */
int TVM_RT_WASM_DefaultModuleGetFunction(Module *mod, const char *func_name, int query_imports,
                                         PackedFunction **out) {
    int status = -1;
    if (mod->module_funcs_map != NULL) {
        status =
            TVM_RT_WASM_TrieQuery(mod->module_funcs_map, (const uint8_t *)func_name, (void **)out);
        if (likely(status != TRIE_NOT_FOUND)) {
            return status;
        }
    }

    if (query_imports) {
        if (mod->env_funcs_map != NULL) {
            status =
                TVM_RT_WASM_TrieQuery(mod->env_funcs_map, (const uint8_t *)func_name, (void **)out);
        }

        if (status && mod->imports) {
            for (size_t i = 0; i < mod->num_imports; ++i) {
                Module *m = mod->imports[i];
                if (m) {
                    status = m->GetFunction(m, func_name, query_imports, out);
                    if (status == 0) {
                        return status;
                    }
                }
            }
        }
    }

    return status;
}

/**
 * @brief Create a module instance from the byte stream.
 * @param type_key The module type key to read.
 * @param type_key_size The module type key string length.
 * @param reader The module binary reader.
 * @param out The pointer to save created module instance.
 * @return 0 if successful
 * @note This function cannot create Library module, such as system library and shared library.
 */
static int TVM_RT_WASM_ModuleCreateFromReader(const char *type_key, size_t type_key_size,
                                              BinaryReader *reader, Module **out) {
    switch (type_key_size) {
    case 4:
        if (!memcmp(type_key, "cuda", 4)) {
            return TVM_RT_WASM_CUDAModuleCreate(reader, out);
        }
        break;
    case 6:
        if (!memcmp(type_key, "webgpu", 6)) {
            return TVM_RT_WASM_WebGPUModuleCreate(reader, out);
        }
        break;
    case 15:
        if (!memcmp(type_key, "metadata_module", 15)) {
            // empty module
            *out = NULL;
            return 0;
        }
        break;
    case 16:
        if (!memcmp(type_key, "relax.Executable", 16)) {
            return TVM_RT_WASM_RelaxExecutableModuleCreate(reader, out);
        }
        break;
    default:
        break;
    }
    TVM_RT_SET_ERROR_RETURN(-1, "Unsupported module type %s", type_key);
}

int TVM_RT_WASM_LibraryModuleLoadBinaryBlob(const char *blob, Module **lib_module) {
    size_t blob_size = (size_t) * (uint64_t *)blob;
    blob += sizeof(uint64_t);

    BinaryReader reader_st = TVM_RT_WASM_BinaryReaderCreate(blob, blob_size);
    if (unlikely(reader_st.current_ptr == NULL)) {
        return -1;
    }
    BinaryReader *reader = &reader_st;
    const char *cur_ptr;
    Module **modules = NULL;
    uint64_t *import_tree_row_ptr = NULL;
    uint64_t *import_tree_child_indices = NULL;
    size_t num_modules = 0;
    size_t num_import_tree_row_ptr = 0;
    size_t num_import_tree_child_indices = 0;
    int status = 0;

#define ModuleBinaryCheckReadOrGoto(_ptr, _read_size)                                              \
    TVM_RT_WASM_BinaryCheckReadOrGoto(_ptr, _read_size, parse_binary_return)

    ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
    size_t key_num = (size_t) * (uint64_t *)cur_ptr;
    modules = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * key_num);
    for (size_t i = 0; i < key_num; ++i) {
        ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
        size_t type_key_size = (size_t) * (uint64_t *)cur_ptr;
        const char *type_key;
        ModuleBinaryCheckReadOrGoto(type_key, type_key_size);

        if (type_key_size == 4 && !memcmp(type_key, "_lib", type_key_size)) {
            modules[num_modules++] = *lib_module;
        } else if (type_key_size == 12 && !memcmp(type_key, "_import_tree", type_key_size)) {
            ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
            num_import_tree_row_ptr = (size_t) * (uint64_t *)cur_ptr;
            if (import_tree_row_ptr == NULL) {
                size_t byte_size = sizeof(uint64_t) * num_import_tree_row_ptr;
                import_tree_row_ptr = TVM_RT_WASM_HeapMemoryAlloc(byte_size);
                ModuleBinaryCheckReadOrGoto(cur_ptr, byte_size);
                memcpy(import_tree_row_ptr, cur_ptr, byte_size);
            }

            ModuleBinaryCheckReadOrGoto(cur_ptr, sizeof(uint64_t));
            num_import_tree_child_indices = (size_t) * (uint64_t *)cur_ptr;
            if (import_tree_child_indices == NULL) {
                size_t byte_size = sizeof(uint64_t) * num_import_tree_child_indices;
                import_tree_child_indices = TVM_RT_WASM_HeapMemoryAlloc(byte_size);
                ModuleBinaryCheckReadOrGoto(cur_ptr, byte_size);
                memcpy(import_tree_child_indices, cur_ptr, byte_size);
            }
        } else {
            status = TVM_RT_WASM_ModuleCreateFromReader(type_key, type_key_size, reader,
                                                        modules + num_modules);
            if (unlikely(status)) {
                goto parse_binary_return;
            }
            ++num_modules;
        }
    }

    if (import_tree_row_ptr == NULL) { // no _import_tree, will no _lib
        (*lib_module)->imports = modules;
        modules = NULL;
        (*lib_module)->num_imports = num_modules;
        if ((*lib_module)->env_funcs_map == NULL) {
            TVM_RT_WASM_TrieCreate(&(*lib_module)->env_funcs_map);
        }
    } else {
        for (size_t i = 0; i < num_modules; ++i) {
            if (unlikely(i + 1 >= num_import_tree_row_ptr ||
                         import_tree_row_ptr[i] > import_tree_row_ptr[i + 1])) {
                break;
            }

            size_t num_imports = (size_t)(import_tree_row_ptr[i + 1] - import_tree_row_ptr[i]);
            if (modules[i] == NULL) { // empty module, such as metadata_module
                if (num_imports == 1) {
                    modules[i] = modules[import_tree_child_indices[import_tree_row_ptr[i]]];
                    continue;
                } else {
                    // todo
                    continue;
                }
            }
            modules[i]->num_imports = num_imports;
            if (num_imports == 0) {
                continue;
            }
            modules[i]->imports = TVM_RT_WASM_HeapMemoryAlloc(sizeof(Module *) * num_imports);
            memset(modules[i]->imports, 0, sizeof(Module *) * num_imports);

            for (uint32_t j = import_tree_row_ptr[i], x = 0; j < import_tree_row_ptr[i + 1];
                 ++j, x++) {
                if (unlikely(j >= num_import_tree_child_indices)) {
                    break;
                }
                modules[i]->imports[x] = modules[import_tree_child_indices[j]];
            }
        }
        // lib_module will be the root in import tree
        *lib_module = modules[0];
        if ((*lib_module)->env_funcs_map == NULL) {
            TVM_RT_WASM_TrieCreate(&(*lib_module)->env_funcs_map);
        }
    }

parse_binary_return:
    if (modules) {
        TVM_RT_WASM_HeapMemoryFree(modules);
    }
    if (import_tree_row_ptr) {
        TVM_RT_WASM_HeapMemoryFree(import_tree_row_ptr);
    }
    if (import_tree_child_indices) {
        TVM_RT_WASM_HeapMemoryFree(import_tree_child_indices);
    }
    return status;
}
