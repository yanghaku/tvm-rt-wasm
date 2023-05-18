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

/*
class Device = GPUDevice;

class Memory {
    dev: GPUDevice;
    buffer: GPUBuffer;
    size: number;
}

class Function {
    dev: GPUDevice;
    pipeline: GPUComputePipeline;
    bind_group_layout: GPUBindGroupLayout;
    bind_group_entries : Array<GPUBindGroupEntry>;
}
*/

EM_ASYNC_JS(int, WGPU_DeviceGet, (WGPU_Device * dev_id_ptr), {
    if (globalThis.TVM_RT_WASM_WEBGPU_CTX == undefined) {
        const err_f = function(msg) {
            console.error(msg);
            _TVMAPISetLastError(stringToUTF8OnStack(msg));
        };
        globalThis.TVM_RT_WASM_WEBGPU_CTX = {
            dev_ids : 0,
            mem_ids : 0,
            func_ids : 0,
            devs : new Map(),
            mems : new Map(),
            funcs : new Map(),
            err_f : err_f,
        };
    }
    const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;

    const adapter_opt = {
        "powerPreference" : "high-performance",
        "forceFallbackAdapter" : false,
    };

    var adapter;
    if (typeof navigator == 'undefined') { // nodejs
        let dawn_path = process.env.DAWN_NODE_PATH;
        if (typeof dawn_path == 'undefined') {
            // todo: search in PATH
            dawn_path = "./dawn.node"; // default use current path
        }
        const dawn = require(dawn_path);
        const gpu = await dawn.create([]);
        adapter = await gpu.requestAdapter(adapter_opt);
    } else { // browser
        if (!('gpu' in navigator)) {
            ctx.err_f('WebGPU not available on this browser (navigator.gpu is not available)');
            return -1;
        }
        adapter = await navigator.gpu.requestAdapter(adapter_opt);
    }

    if (adapter == null || adapter == undefined || adapter == 'undefined') {
        ctx.err_f('No Adapter found');
        return -1;
    }

    const limits = adapter.limits;
    const dev = await adapter.requestDevice({
        requiredLimits : {
            maxBindGroups : limits.maxBindGroups,
            maxBindingsPerBindGroup : limits.maxBindingsPerBindGroup,
            maxBufferSize : limits.maxBufferSize,
            maxComputeInvocationsPerWorkgroup : limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX : limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY : limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ : limits.maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupStorageSize : limits.maxComputeWorkgroupStorageSize,
            maxComputeWorkgroupsPerDimension : limits.maxComputeWorkgroupsPerDimension,
            maxDynamicStorageBuffersPerPipelineLayout : limits.maxDynamicStorageBuffersPerPipelineLayout,
            maxDynamicUniformBuffersPerPipelineLayout : limits.maxDynamicUniformBuffersPerPipelineLayout,
            maxSamplersPerShaderStage : limits.maxSamplersPerShaderStage,
            maxStorageBufferBindingSize : limits.maxStorageBufferBindingSize,
            maxStorageBuffersPerShaderStage : limits.maxStorageBuffersPerShaderStage,
            maxStorageTexturesPerShaderStage : limits.maxStorageTexturesPerShaderStage,
            minStorageBufferOffsetAlignment : limits.minStorageBufferOffsetAlignment,
        }
    });

    ctx.dev_ids += 1;
    const dev_id = ctx.dev_ids;
    ctx.devs.set(dev_id, dev);
    setValue(dev_id_ptr, dev_id, '*');
    return 0;
});

EM_JS(int, WGPU_DeviceFree, (WGPU_Device dev_id), {
    const devs = globalThis.TVM_RT_WASM_WEBGPU_CTX.devs;
    const dev = devs.get(dev_id);
    if (dev != undefined) {
        devs.delete(dev_id);
        dev.destroy();
    }
    return 0;
});

EM_JS(int, WGPU_MemoryAlloc, (WGPU_Device dev_id, WGPU_Memory *mem_id_ptr, size_t nbytes), {
    const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
    const dev = ctx.devs.get(dev_id);
    if (nbytes & 3) {
        nbytes = (nbytes | 3) + 1;
    }
    const buffer = dev.createBuffer({
        size : nbytes,
        usage : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    ctx.mem_ids += 1;
    const mem_id = ctx.mem_ids;
    ctx.mems.set(mem_id, {
        dev : dev,
        buffer : buffer,
        size : nbytes,
    });
    setValue(mem_id_ptr, mem_id, '*');
    return 0;
});

EM_JS(int, WGPU_MemoryFree, (WGPU_Memory mem_id), {
    const mems = globalThis.TVM_RT_WASM_WEBGPU_CTX.mems;
    const mem = mems.get(mem_id);
    if (mem != undefined) {
        mems.delete(mem_id);
        mem.buffer.destroy();
    }
    return 0;
});

EM_JS(int, WGPU_MemoryCopyHtoD,
      (WGPU_Memory dst, size_t dst_byte_offset, void *src, size_t src_byte_offset, size_t nbytes), {
          const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
          const dst_mem = ctx.mems.get(dst);
          dst_mem.dev.queue.writeBuffer(dst_mem.buffer, dst_byte_offset, HEAPU8, src + src_byte_offset, nbytes);
          return 0;
      });

EM_ASYNC_JS(int, WGPU_MemoryCopyDtoH,
            (void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes), {
                const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
                const src_mem = ctx.mems.get(src);
                const dev = src_mem.dev;

                let map_size; /* must %4==0 */
                if (nbytes & 3 != 0) {
                    map_size = (nbytes | 3) + 1;
                } else {
                    map_size = nbytes;
                }
                const dst_gpu = dev.createBuffer({
                    size : map_size,
                    usage : GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                });

                const cmd_encoder = dev.createCommandEncoder();
                cmd_encoder.copyBufferToBuffer(src_mem.buffer, src_byte_offset, dst_gpu, 0, nbytes);
                dev.queue.submit([cmd_encoder.finish()]);

                await dst_gpu.mapAsync(GPUMapMode.READ, 0, map_size);
                HEAPU8.set(new Uint8Array(dst_gpu.getMappedRange(0, map_size)), dst + dst_byte_offset);
                dst_gpu.unmap();
                dst_gpu.destroy();
                return 0;
            });

EM_JS(int, WGPU_MemoryCopyDtoD,
      (WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes), {
          const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
          const dst_mem = ctx.mems.get(dst);
          const src_mem = ctx.mems.get(src);
          const dev = dst_mem.dev;
          const cmd_encoder = dev.createCommandEncoder();
          cmd_encoder.copyBufferToBuffer(src_mem.buffer, src_byte_offset, dst_mem.buffer, dst_byte_offset, nbytes);
          dev.queue.submit([cmd_encoder.finish()]);
          return 0;
      });

EM_JS(int, WGPU_FunctionCreate,
      (WGPU_Device dev_id, WGPU_Function *func_id_ptr, const char *s, uint32_t s_len, const char *e, uint32_t e_len,
       uint32_t num_args),
      {
          const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
          const dev = ctx.devs.get(dev_id);

          const bind_group_layout_entries = [];
          for (let i = 0; i < num_args; ++i) {
              bind_group_layout_entries.push({
                  binding : i,
                  visibility : GPUShaderStage.COMPUTE,
                  buffer : {
                      type : "storage",
                  },
              });
          }

          const bind_group_layout = dev.createBindGroupLayout({entries : bind_group_layout_entries});
          const layout = dev.createPipelineLayout({bindGroupLayouts : [bind_group_layout]});

          const module = dev.createShaderModule({
              code : UTF8ToString(s, s_len),
              hints : {
                  main : {layout : layout},
              }
          });
          const pipeline = dev.createComputePipeline({
              layout : layout,
              compute : {
                  module : module,
                  entryPoint : "main",
              }
          });

          const bind_group_entries = [];
          for (let i = 0; i < num_args; ++i) {
              bind_group_entries.push({
                  binding : i,
              });
          }

          ctx.func_ids += 1;
          const func_id = ctx.func_ids;
          ctx.funcs.set(func_id, {
              dev : dev,
              pipeline : pipeline,
              bind_group_layout : bind_group_layout,
              bind_group_entries : bind_group_entries,
          });
          setValue(func_id_ptr, func_id, '*');
          return 0;
      });

EM_JS(int, WGPU_FunctionRun,
      (WGPU_Function func_id, const WGPU_Memory *args, uint32_t num_args, size_t grid_x, size_t grid_y, size_t grid_z),
      {
          const ctx = globalThis.TVM_RT_WASM_WEBGPU_CTX;
          const func = ctx.funcs.get(func_id);
          const dev = func.dev;
          for (let i = 0; i < num_args; ++i) {
              const mem_id = getValue(args, '*');
              func.bind_group_entries[i].resource = {
                  buffer : ctx.mems.get(mem_id).buffer,
              };
              // now the pointer size is 32bit
              args += 4;
          }

          const cmd_encoder = dev.createCommandEncoder();
          const compute = cmd_encoder.beginComputePass();
          compute.setPipeline(func.pipeline);

          compute.setBindGroup(0, dev.createBindGroup({
              layout : func.bind_group_layout,
              entries : func.bind_group_entries,
          }));
          compute.dispatchWorkgroups(grid_x, grid_y, grid_z);
          compute.end();
          dev.queue.submit([cmd_encoder.finish()]);
          return 0;
      });

EM_JS(int, WGPU_FunctionFree, (WGPU_Function func_id), {
    const funcs = globalThis.TVM_RT_WASM_WEBGPU_CTX.funcs;
    const func = funcs.get(func_id);
    if (func != undefined) {
        funcs.delete(func_id);
        func.bind_group_layout.destroy();
        func.pipeline.destroy();
    }
    return 0;
});

#endif // USE_WEBGPU = 1 && defined(__EMSCRIPTEN__)
