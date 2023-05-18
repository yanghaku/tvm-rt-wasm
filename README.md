<div style="text-align: center">
	<h1>TVM Runtime for WebAssembly</h1>
	<p>
    <a href="https://github.com/yanghaku/tvm-rt-wasm/blob/main/LICENSE">
	    <img src="https://img.shields.io/badge/license-Apache-brightgreen" alt="License">
    </a>
	</p>
</div>
<hr/>

A High performance and tiny [TVM] graph executor library written in C which can compile to WebAssembly and use CUDA/WebGPU as the backend.

## Support Matrix

| Toolchain  | Target                      | Backend    | Runtime                                    |
| ---------- | --------------------------- | ---------- | ------------------------------------------ |
| wasi-sdk   | WebAssembly                 | CPU        | wasmer, WasmEdge, wasmtime, etc.           |
| wasi-sdk   | WebAssembly                 | **CUDA**   | wasmer-gpu (not open-sourced now)          |
| emscripten | WebAssembly                 | CPU        | browser, nodejs                            |
| emscripten | WebAssembly                 | **WebGPU** | **browser**, **nodejs** (need `dawn.node`) |
| clang/gcc  | native(x86_64,aarch64,etc.) | CPU        | /                                          |
| clang/gcc  | native(x86_64,aarch64,etc.) | CUDA       | /                                          |
| clang/gcc  | native(x86_64,aarch64,etc.) | WebGPU     | link with use [dawn] or [webgpu-native]    |

## Build the library

### Requirements:

1. cmake
2. C compiler (clang/gcc/msvc for native target, wasi-sdk/emscripten for WebAssembly target)

WebAssembly target toolchain download: [wasi-sdk github repo], [emsdk github repo]

### Available Options in cmake

| Variable        | Default       | Description                                                          |
| --------------- | ------------- | -------------------------------------------------------------------- |
| USE_EMSDK       | OFF           | Use emsdk toolchain and compile to target ```(wasm32-emscription)``` |
| USE_WASI_SDK    | OFF           | Use wasi-sdk toolchain and compile to target ```(wasm32-wasi)```     |
| USE_CUDA        | OFF           | Use CUDA support                                                     |
| USE_WEBGPU      | OFF           | Use WebGPU support                                                   |
| WASI_SDK_PREFIX | /opt/wasi-sdk | The path to wasi-sdk                                                 |
| EMSDK_PREFIX    | /opt/emsdk    | The path to emsdk                                                    |

### Do build

Sample: build the ```wasm32-wasi``` target with CUDA support, the target can run with [**```wasmer-gpu```**].

```shell
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_WASI_SDK=ON
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
