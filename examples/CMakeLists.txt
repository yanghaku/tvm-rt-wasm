cmake_minimum_required(VERSION 3.19)

# config for examples

# wasi-sdk
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX D:/lib/wasi-sdk)
endif ()

# cuda home
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT D:/lib/cuda/)
endif ()

# tvm-rt-wasm installed path
if (NOT DEFINED TVM_RT_WASM_ROOT)
    set(TVM_RT_WASM_ROOT D:/lib/tvm-rt-wasm/)
endif ()

# set the ar path
set(AR_EXEC llvm-ar)
OPTION(USE_WASI_SDK "USE WASI SDK" ON)
option(EXAMPLE_USE_CUDA "example use cuda" ON)


message("examples USE WASI-SDK = ${USE_WASI_SDK}")
if (${USE_WASI_SDK})
    message("examples WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(PY_RUNTIME wasm)
else ()
    set(PY_RUNTIME native)
endif ()

# whether use cuda
message("examples USE_CUDA = ${EXAMPLE_USE_CUDA}")
if (${EXAMPLE_USE_CUDA})
    message("examples CUDAToolkit path = ${CUDAToolkit_ROOT}")
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    if (NOT ${USE_WASI_SDK})
        link_directories(${CUDAToolkit_ROOT}/lib64)
        link_libraries(cuda)
    else ()
        add_link_options(-Wl,--allow-undefined)
    endif ()
    SET(PY_TARGET cuda)
else ()
    SET(PY_TARGET cpu)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-examples)

add_compile_options(-fno-exceptions -O3 -w -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-function -fvisibility-inlines-hidden -flto=full)

include_directories(${TVM_RT_WASM_ROOT}/include)
link_directories(${TVM_RT_WASM_ROOT}/lib)

find_package(Python3 COMPONENTS Interpreter)

# mobilenet0.25
set(mobilenet0.25_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet0.25/)
set(mobilenet0.25.tvm_gen ${mobilenet0.25_lib_dir}/graph.json.c ${mobilenet0.25_lib_dir}/graph.params.c)
set(mobilenet0.25.tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/scripts/mobilenet0.25.py)
add_custom_command(
        OUTPUT ${mobilenet0.25.tvm_gen}
        COMMAND python ${mobilenet0.25.tvm_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${mobilenet0.25_lib_dir}
        COMMAND tar -xf ${mobilenet0.25_lib_dir}/graph.tar -C ${mobilenet0.25_lib_dir}
        COMMAND ${AR_EXEC} -r ${mobilenet0.25_lib_dir}/graph.lib ${mobilenet0.25_lib_dir}/*.o
        COMMAND ${AR_EXEC} -r ${mobilenet0.25_lib_dir}/libgraph.a ${mobilenet0.25_lib_dir}/*.o
        DEPENDS ${mobilenet0.25.tvm_script}
        VERBATIM
)
add_executable(mobilenet0.25.wasm mobilenet0.25/mobilenet0.25.c ${mobilenet0.25.tvm_gen})
target_link_libraries(mobilenet0.25.wasm tvm-rt)
target_link_directories(mobilenet0.25.wasm PRIVATE ${mobilenet0.25_lib_dir})
target_link_libraries(mobilenet0.25.wasm graph)
