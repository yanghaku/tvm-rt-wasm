<div style="text-align: center">
	<h1>TVM Runtime for WebAssembly</h1>
	<p>
    <a href="https://github.com/yanghaku/tvm-rt-wasm/blob/main/LICENSE">
	    <img src="https://img.shields.io/badge/license-Apache-brightgreen" alt="License">
    </a>
	</p>
</div>
<hr/>

A High performance and tiny tvm graph executor library written in C which can enable cuda and compile to WebAssembly.

implement the api for ```tvm/runtime/c_runtime_api.h``` and ```tvm/runtime/c_backend_api.h```.

## Build the library

### Requirements:

1. cmake
2. C compiler (clang/gcc/msvc for native target, wasi-sdk/emscripten for WebAssembly target)

WebAssembly target toolchain download: [wasi-sdk github repo], [emsdk github repo]

### Available Options in cmake

| Variable        | Default       | Description                                                          |
|-----------------|---------------|----------------------------------------------------------------------|
| USE_EMSDK       | OFF           | Use emsdk toolchain and compile to target ```(wasm32-emscription)``` |
| USE_WASI_SDK    | OFF           | Use wasi-sdk toolchain and compile to target ```(wasm32-wasi)```     |
| USE_CUDA        | ON            | Use CUDA support                                                     |
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

## Use the library

See the [examples](./examples)


<!-- some external links-->

[wasi-sdk github repo]: https://github.com/WebAssembly/wasi-sdk

[tvm install tutorial]: https://tvm.apache.org/docs/tutorial/install.html#installing-from-binary-packages

[tvm github repo]: https://github.com/apache/tvm/

[**```wasmer-gpu```**]: https://github.com/yanghaku/wasmer-gpu

[```tlcpack```]: https://tlcpack.ai/

[emsdk github repo]: https://github.com/emscripten-core/emsdk
