cmake_minimum_required(VERSION 3.19)

# config for examples

# wasi-sdk
set(WASI_SDK_PREFIX D:/lib/wasi-sdk)

# if use cuda
set(EXAMPLE_USE_CUDA OFF)

# tvm-rt-wasm installed path
set(TVM_RT_WASM_ROOT D:/lib/tvm-rt-wasm/)

# set the ar path
set(AR_EXEC llvm-ar)

#-----------------------------------------------------------------------------------------------------------------------

# wasi-sdk
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)

project(tvm-rt-examples)
message("examples for tvm-rt")

add_compile_options(-fno-exceptions -O3 -Wall -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-function -fvisibility-inlines-hidden -fno-rtti)
add_link_options(-Wl,--allow-undefined)

message("EXAMPLE_USE_CUDA = ${EXAMPLE_USE_CUDA}")

include_directories(${TVM_RT_WASM_ROOT}/include)
link_directories(${TVM_RT_WASM_ROOT}/lib)

if (${EXAMPLE_USE_CUDA})
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    SET(PY_TARGET cuda)
else ()
    SET(PY_TARGET cpu)
endif ()

find_package(Python3 COMPONENTS Interpreter)

# mobilenet0.25
set(mobilenet0.25_lib_dir ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/lib)
set(mobilenet0.25.tvm_gen ${mobilenet0.25_lib_dir}/graph.json.c ${mobilenet0.25_lib_dir}/graph.params.c)
set(mobilenet0.25.tvm_script ${mobilenet0.25_lib_dir}/../scripts/mobilenet0.25.py)
add_custom_command(
        OUTPUT ${mobilenet0.25.tvm_gen}
        COMMAND python ${mobilenet0.25.tvm_script} --target ${PY_TARGET} -o ${mobilenet0.25_lib_dir}
        COMMAND tar -xf ${mobilenet0.25_lib_dir}/graph.tar -C ${mobilenet0.25_lib_dir}
        COMMAND ${AR_EXEC} -r ${mobilenet0.25_lib_dir}/libgraph.a ${mobilenet0.25_lib_dir}/*.o
        DEPENDS ${mobilenet0.25.tvm_script}
        VERBATIM
)
add_executable(mobilenet0.25.wasm mobilenet0.25/mobilenet0.25.c ${mobilenet0.25.tvm_gen})
target_link_libraries(mobilenet0.25.wasm tvm-rt)
target_link_directories(mobilenet0.25.wasm PRIVATE ${mobilenet0.25_lib_dir})
target_link_libraries(mobilenet0.25.wasm graph)
