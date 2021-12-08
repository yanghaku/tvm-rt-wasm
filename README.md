# TVM Runtime for WebAssembly

## Build for WebAssembly

requirements:

1. cmake and ninja
2. tvm python package (http://210.28.132.171/yangbo/tvm-src)
3. wasi-sdk
4. cuda toolkit (cuda version >= 11.2)

The wasi-sdk and cuda-toolkit path, you can modify in the CmakeLists.txt or use command line.

### CUDA Version

the option ```USE_CUDA``` is default ON, so you just use cmake to

```shell
mkdir -p build
cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release [ -DWASI_SDK_PREFIX=/wasi-sdk/path -DCUDAToolkit_ROOT=/cuda/path ]
```

the example for full options is that:

```shell
cmake -DCMAKE_BUILD_TYPE=Release -G "Ninja" .. -DUSE_CUDA=ON -DEXAMPLE_USE_CUDA=ON -DUSE_WASI_SDK=OFF -DWASI_SDK_PREFIX=/opt/wasi-sdk -DCUDAToolkit_ROOT=/usr/local/cuda
```

build the static library:

```shell
ninja
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


## Run the example

if you build the examples into WebAssembly, for cpu only, you can run it using any WASM-runtime, but **for CUDA version**, you must use **wasmer-gpu**.

the **wasmer-gpu** is in: http://210.28.132.171/yangbo/wasmer-gpu

