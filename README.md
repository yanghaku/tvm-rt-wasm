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

## Build for WebAssembly

requirements:

1. cmake and ninja
2. tvm python
   package ([tvm github repo]) ([tvm install tutorial])
3. wasi-sdk-14.0 ([wasi-sdk github repo])
4. cuda toolkit (cuda version >= 10.2.89)

The wasi-sdk and cuda-toolkit path, you can modify in the CmakeLists.txt or use command line to specify.

build examples python package requirements:

* tvm (or [```tlcpack```])
* mxnet (for example mobilenet)
* onnx (for example resnet,vgg)
* pillow

### CUDA Version

the option ```USE_CUDA``` is default ```ON```, so you just use cmake to build static library

```shell
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release [ -DWASI_SDK_PREFIX=/wasi-sdk/path -DCUDAToolkit_ROOT=/cuda/path ]
ninja
```

the example for full options is that:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -G "Ninja" .. -DUSE_CUDA=ON -DEXAMPLE_USE_CUDA=ON -DUSE_WASI_SDK=OFF -DWASI_SDK_PREFIX=/opt/wasi-sdk -DCUDAToolkit_ROOT=/usr/local/cuda
```

you also can build the examples: (build examples need tvm python environment to general modules).

```shell
ninja mobilenet0.25.wasm
```

### CPU-only Version

if you want build cpu version, you can set ```USE_CUDA=OFF```, and for examples you can set ```EXAMPLE_USE_CUDA=OFF```.

```shell
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DEXAMPLE_USE_CUDA=OFF [ -DWASI_SDK_PREFIX=/wasi-sdk/path ]
ninja
ninja mobilenet0.25.wasm
```

## build for native

If you want to build for native to run, you can just add the option  ```USE_WASI_SDK=OFF```
(now only test ```gcc``` in linux)

## Run the example

if you build the examples into WebAssembly, for cpu only, you can run it using any WASM-runtime, but **for CUDA
version**, you must use [**```wasmer-gpu```**].


<!-- some external links-->

[wasi-sdk github repo]: https://github.com/WebAssembly/wasi-sdk

[tvm install tutorial]: https://tvm.apache.org/docs/tutorial/install.html#installing-from-binary-packages

[tvm github repo]: https://github.com/apache/tvm/

[**```wasmer-gpu```**]: https://github.com/yanghaku/wasmer-gpu

[```tlcpack```]: https://tlcpack.ai/
