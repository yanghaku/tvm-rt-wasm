/*!
 * \file src/runtime/module/module.c
 * \brief implement functions for module.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module/cuda_module.h>
#include <tvm/runtime/module/module.h>
#include <tvm/runtime/utils/common.h>

/*!
 * \brief the symbols for system library
 * \note this Trie will be managed by sys_lib_module instance after init
 */
Trie *system_lib_symbol = NULL;

/*! \brief the system library module is a single instance */
static Module *sys_lib_module = NULL;

/*!
 * \brief Import another module into this module.
 * \param other The module to be imported.
 *
 * \note Cyclic dependency is not allowed among modules,
 *  An error will be thrown when cyclic dependency is detected.
 */
int ModuleImport(Module *self, Module *other) { return -1; }

/*! \brief the simple release function for system_lib_module and dso_lib_module */
static int DefaultModuleReleaseFunc(Module *self) {
    DLDevice cpu = {kDLCPU, 0};
    if (self->imports) {
        for (uint32_t i = 0; i < self->num_imports; ++i) {
            self->imports[i]->Release(self->imports[i]);
        }
        TVMDeviceFreeDataSpace(cpu, self->imports);
    }
    if (self->module_funcs_map) {
        TrieRelease(self->module_funcs_map);
    }
    if (self->env_funcs_map) {
        TrieRelease(self->env_funcs_map);
    }
    return TVMDeviceFreeDataSpace(cpu, self);
}

/*!
 * \brief Load from binary blob
 * @param blob the dev_blob binary
 * @param lib_module the root module handle
 * @return 0 if successful
 */
static int ModuleLoadBinaryBlob(const char *blob, Module **lib_module) {
    DLDevice cpu = {kDLCPU, 0};
    DLDataType dataType = {0, 0, 0};

    //    uint64_t blob_size = *(uint64_t *)blob;
    blob += sizeof(uint64_t);

    uint32_t key_num = (uint32_t) * (uint64_t *)blob;
    blob += sizeof(uint64_t);

    Module **modules = NULL;
    TVMDeviceAllocDataSpace(cpu, sizeof(Module *) * key_num, 0, dataType, (void **)&modules);
    uint64_t *import_tree_row_ptr = NULL;
    uint64_t *import_tree_child_indices = NULL;
    uint32_t num_modules = 0;
    uint32_t num_import_tree_row_ptr = 0;
    uint32_t num_import_tree_child_indices = 0;
    int status;

    for (uint32_t i = 0; i < key_num; ++i) {
        uint32_t mod_type_key_size = (uint32_t) * (uint64_t *)blob;
        blob += sizeof(uint64_t);

        if (mod_type_key_size == 4 && !memcmp(blob, "_lib", mod_type_key_size)) {
            modules[num_modules++] = *lib_module;
            blob += mod_type_key_size;
        } else if (mod_type_key_size == 12 && !memcmp(blob, "_import_tree", mod_type_key_size)) {
            blob += mod_type_key_size;

            num_import_tree_row_ptr = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t);
            TVMDeviceAllocDataSpace(cpu, sizeof(uint64_t) * num_import_tree_row_ptr, 0, dataType,
                                    (void **)&import_tree_row_ptr);
            memcpy(import_tree_row_ptr, blob, sizeof(uint64_t) * num_import_tree_row_ptr);
            blob += sizeof(uint64_t) * num_import_tree_row_ptr;

            num_import_tree_child_indices = (uint32_t) * (uint64_t *)blob;
            blob += sizeof(uint64_t);
            TVMDeviceAllocDataSpace(cpu, sizeof(uint64_t) * num_import_tree_child_indices, 0, dataType,
                                    (void **)&import_tree_child_indices);
            memcpy(import_tree_child_indices, blob, sizeof(uint64_t) * num_import_tree_child_indices);
            blob += sizeof(uint64_t) * num_import_tree_child_indices;

        } else {
            const char *key = blob;
            blob += mod_type_key_size;
            status = ModuleFactory(key, blob, MODULE_FACTORY_RESOURCE_BINARY, &modules[num_modules]);
            if (unlikely(status <= 0)) { // ModuleFactory will return offset
                return status;
            }
            blob += status;
            ++num_modules;
        }
    }

    if (import_tree_row_ptr == NULL) { // no _import_tree, will no _lib
        (*lib_module)->allocated_imports_size = key_num;
        (*lib_module)->imports = modules;
        (*lib_module)->num_imports = num_modules;
        // cache all env function
        for (uint32_t i = 0; i < num_modules; ++i) {
            if (modules[i]->module_funcs_map) {
                TrieInsertAll((*lib_module)->env_funcs_map, modules[i]->module_funcs_map);
            }
        }
    } else {
        for (uint32_t i = 0; i < num_modules; ++i) {
            if (unlikely(i + 1 >= num_import_tree_row_ptr || import_tree_row_ptr[i] > import_tree_row_ptr[i + 1])) {
                break;
            }

            modules[i]->num_imports = (uint32_t)(import_tree_row_ptr[i + 1] - import_tree_row_ptr[i]);
            modules[i]->allocated_imports_size = modules[i]->num_imports;
            if (modules[i]->num_imports == 0) {
                continue;
            }
            TVMDeviceAllocDataSpace(cpu, sizeof(Module *) * modules[i]->num_imports, 0, dataType,
                                    (void **)&modules[i]->imports);

            for (uint32_t j = import_tree_row_ptr[i], x = 0; j < import_tree_row_ptr[i + 1]; ++j, x++) {
                if (unlikely(j >= num_import_tree_child_indices)) {
                    break;
                }
                modules[i]->imports[x] = modules[import_tree_child_indices[j]];
            }
        }
        // lib_module will be the root in import tree
        *lib_module = modules[0];
        for (uint32_t i = 1; i < num_modules; ++i) {
            if (modules[i]->module_funcs_map) {
                TrieInsertAll((*lib_module)->env_funcs_map, modules[i]->module_funcs_map);
            }
        }
        TVMDeviceFreeDataSpace(cpu, modules);
        TVMDeviceFreeDataSpace(cpu, import_tree_row_ptr);
        TVMDeviceFreeDataSpace(cpu, import_tree_child_indices);
    }
    return 0;
}

/*!
 * \brief create a system library module (this will be a single instance)
 * @param libraryModule the out handle
 * @return 0 if successful
 */
static int SystemLibraryModuleCreate(Module **libraryModule) {
    if (likely(sys_lib_module != NULL)) { // if the instance exists, return
        *libraryModule = sys_lib_module;
        return 0;
    }
    if (unlikely(system_lib_symbol == NULL)) {
        SET_ERROR_RETURN(-1, "no symbol in system library!");
    }

    DLDevice cpu = {kDLCPU, 0};
    DLDataType no_type = {0, 0, 0};
    TVMDeviceAllocDataSpace(cpu, sizeof(Module), 0, no_type, (void **)libraryModule);
    memset(*libraryModule, 0, sizeof(Module));

    (*libraryModule)->Release = DefaultModuleReleaseFunc;
    (*libraryModule)->module_funcs_map = system_lib_symbol;
    TrieCreate(&(*libraryModule)->env_funcs_map);
    system_lib_symbol = NULL; // manage this Trie*

    // dev_blob
    const char *blob = NULL;
    int status = TrieQuery((*libraryModule)->module_funcs_map, (const uint8_t *)TVM_DEV_MODULE_BLOB, (void **)&blob);
    if (status == TRIE_SUCCESS) {
        status = ModuleLoadBinaryBlob(blob, libraryModule);
        if (unlikely(status)) {
            return status;
        }
    }

    // module context
    void **module_context = NULL;
    status = TrieQuery((*libraryModule)->module_funcs_map, (const uint8_t *)TVM_MODULE_CTX, (void **)&module_context);
    if (likely(status == TRIE_SUCCESS)) {
        *module_context = *libraryModule;
    }

    sys_lib_module = *libraryModule;
    return status;
}

/*!
 * \brief create a library module from the dynamic shared library
 * @param filename the filename
 * @param libraryModule the out handle
 * @return 0 if successful
 */
static int DSOLibraryModuleCreate(const char *filename, Module **libraryModule) {
    // todo: implement it
    SET_ERROR_RETURN(-1, "now it's unimplemented yet");
}

/*!
 * \brief create a module instance for given type
 * @param type the module type or file format
 * @param resource filename or binary source
 * @param resource_type Specify whether resource is binary or file type;  0: binary 1: file
 * @param out the pointer to receive created instance
 * @return >=0 if successful   (if binary type, it should return the binary length it has read)
 */
int ModuleFactory(const char *type, const char *resource, int resource_type, Module **out) {
    if (!memcmp(type, MODULE_SYSTEM_LIB, strlen(MODULE_SYSTEM_LIB))) {
        return SystemLibraryModuleCreate(out);
    }
    if (!memcmp(type, "so", 2) || !memcmp(type, "dll", 3) || !memcmp(type, "dylib", 5)) {
        if (unlikely(resource_type != MODULE_FACTORY_RESOURCE_FILE)) {
            SET_ERROR_RETURN(-1, "the dso library can only be load from file");
        }
        return DSOLibraryModuleCreate(resource, out);
    }
    if (!memcmp(type, "cuda", 4)) {
        return CUDAModuleCreate(resource, resource_type, (CUDAModule **)out);
    }
    fprintf(stderr, "unsupported module type %s\n", type);
    SET_ERROR_RETURN(-1, "unsupported module type %s\n", type);
}
