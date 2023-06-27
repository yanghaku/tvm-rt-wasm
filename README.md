<div style="text-align: center">
	<h1>TVM Runtime for WebAssembly</h1>
	<p>
    <a href="https://github.com/yanghaku/tvm-rt-wasm/blob/main/LICENSE">
	    <img src="https://img.shields.io/badge/license-Apache-brightgreen" alt="License">
    </a>
	</p>
</div>
<hr/>

A High performance and tiny [TVM] graph executor library written in C which can compile to WebAssembly and use
CUDA/WebGPU as the accelerator.

## Support Matrix

| Toolchain  | Target                      | Backend    | Runtime                                    |
|------------|-----------------------------|------------|--------------------------------------------|
| wasi-sdk   | WebAssembly                 | CPU        | wasmer, WasmEdge, wasmtime, etc.           |
| wasi-sdk   | WebAssembly                 | **CUDA**   | wasmer-gpu (not open-sourced now)          |
| emscripten | WebAssembly                 | CPU        | browser, nodejs                            |
| emscripten | WebAssembly                 | **WebGPU** | **browser**, **nodejs** (need `dawn.node`) |
| clang/gcc  | native(x86_64,aarch64,etc.) | CPU        | /                                          |
| clang/gcc  | native(x86_64,aarch64,etc.) | CUDA       | /                                          |
| clang/gcc  | native(x86_64,aarch64,etc.) | WebGPU     | link with use [dawn] or [webgpu-native]    |

## Build from source.

### Requirements:

1. cmake
2. C compiler (clang/gcc/msvc for native target, wasi-sdk/emscripten for WebAssembly target)

WebAssembly target toolchain download: [wasi-sdk github repo], [emsdk github repo]

### Available Options in cmake

## Project options:

* `USE_WASI_SDK`: If use the wasi-sdk, set `/path/to/wasi-sdk` or set `ON`/`AUTO`/`TRUE` to use default `/opt/wasi-sdk`.
* `USE_EMSDK`: If use the emscripten, set `/path/to/emsdk` or set `ON`/`AUTO`TRUE` to use default `/opt/emsdk`.

### Do build

Sample: build the ```wasm32-wasi``` target with CUDA support, the target can run with [**```wasmer-gpu```**].

```shell
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_WASI_SDK=ON
ninja
```

## Use the library in C

See the [examples](./examples)


<!-- some external links-->

[wasi-sdk github repo]: https://github.com/WebAssembly/wasi-sdk

[tvm install tutorial]: https://tvm.apache.org/docs/tutorial/install.html#installing-from-binary-packages

[TVM]: https://github.com/apache/tvm/

[**```wasmer-gpu```**]: https://github.com/yanghaku/wasmer-gpu

[```tlcpack```]: https://tlcpack.ai/

[emsdk github repo]: https://github.com/emscripten-core/emsdk

[dawn]: https://dawn.googlesource.com/dawn

[webgpu-native]: https://github.com/gfx-rs/wgpu-native
