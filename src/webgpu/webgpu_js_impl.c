/*!
 * \file webgpu/webgpu_js_impl.c
 * \brief implement for webgpu sync c api using inline js to run in nodejs or browser.
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#if USE_WEBGPU && defined(__EMSCRIPTEN__) // USE_WEBGPU = 1 && defined(__EMSCRIPTEN__)

#include <emscripten.h>
#include <tvm/runtime/c_runtime_api.h>
#include <webgpu/webgpu_c_api.h>

struct WGPU_Device_st {};

struct WGPU_Memory_st {};

struct WGPU_Function_st {};

EM_ASYNC_JS(void *, request_adapter, (), {
    const adapter_option = {
        "powerPreference" : "high-performance",
        "forceFallbackAdapter" : false,
    };

    var adapter;
    if (typeof navigator == 'undefined') { // nodejs
        let dawn_path = process.env.DAWN_NODE_PATH;
        if (typeof dawn_path == 'undefined') {
            dawn_path = "./dawn.node"; // default use current path
        }
        const dawn = require(dawn_path);
        const gpu = await dawn.create([]);
        adapter = await gpu.requestAdapter(adapter_option);
    } else { // browser
        if (!('gpu' in navigator)) {
            console.log('WebGPU not available on this browser (navigator.gpu is not available)');
            return NULL;
        }
        adapter = await navigator.gpu.requestAdapter(adapter_option);
    }

    if (adapter == null || adapter == undefined || adapter == 'undefined') {
        console.log('error');
    }
});

static __attribute__((constructor)) void webgpu_adapter_constructor() { void *d = request_adapter(); }

int WGPU_DeviceGet(WGPU_Device *device_ptr) { return 0; }

int WGPU_DeviceFree(WGPU_Device device) { return 0; }

int WGPU_MemoryAlloc(WGPU_Device device, WGPU_Memory *memory_ptr, size_t nbytes) { return 0; }

int WGPU_MemoryFree(WGPU_Memory memory) { return 0; }

int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, void *src, size_t src_byte_offset, size_t nbytes) {
    return 0;
}

int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes) {
    return 0;
}

int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes) {
    return 0;
}

int WGPU_FunctionCreate(WGPU_Function *func_ptr, char *source) { return 0; }

int WGPU_FunctionRun(WGPU_Function function, size_t block_dims[3], size_t thread_dims[3], WGPU_Memory kernel_args[],
                     uint32_t num_kernel_args) {
    return 0;
}

int WGPU_FunctionFree(WGPU_Function function) { return 0; }

#endif // USE_WEBGPU = 1 && defined(__EMSCRIPTEN__)
