cmake_minimum_required(VERSION 3.19)

# config for examples

# wasi-sdk
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()

# cuda home
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT /usr/local/cuda-11.5)
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
    add_link_options(-flto -Wl,--allow-undefined,--lto-O3,--gc-sections,--strip-all)
else ()
    set(PY_RUNTIME native)
    link_directories(${CUDAToolkit_ROOT}/lib64)
    link_directories(${CUDAToolkit_ROOT}/lib64/stubs)
    link_directories(${CUDAToolkit_ROOT}/lib/)
    link_directories(${CUDAToolkit_ROOT}/lib/x64)
    link_libraries(m)

    # check if the linker is lld
    execute_process(COMMAND ${CMAKE_LINKER} --version OUTPUT_VARIABLE LD_VERSION_OUTPUT)
    if (${LD_VERSION_OUTPUT} MATCHES "LLD*")
        # lld cannot link .ctos and .init-array
        add_link_options(-fuse-ld=gold)
    endif ()
endif (${USE_WASI_SDK})

# whether use cuda
message("examples USE_CUDA = ${EXAMPLE_USE_CUDA}")
if (${EXAMPLE_USE_CUDA})
    message("examples CUDAToolkit path = ${CUDAToolkit_ROOT}")
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    if (NOT ${USE_WASI_SDK})
        # build for native need link with cuda library
        link_libraries(cuda)
    endif ()
    SET(PY_TARGET cuda)
else ()
    SET(PY_TARGET cpu)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-examples)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
                -w -Wextra -Werror -Wl,--strip-all,--gc-sections")

include_directories(common_headers)
#include_directories(${TVM_RT_WASM_ROOT}/include)
#link_directories(${TVM_RT_WASM_ROOT}/lib)

find_package(Python3 COMPONENTS Interpreter)


function(custom_tvm_executable target_name source_files tvm_gens tvm_py_script out_lib_dir)
    if (MSVC)
        set(GRAPH_LIB ${out_lib_dir}/graph.lib)
    else ()
        set(GRAPH_LIB ${out_lib_dir}/libgraph.a)
    endif ()
    add_custom_command(
            OUTPUT ${tvm_gens}
            COMMAND python ${tvm_py_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${out_lib_dir}
            COMMAND tar -xf ${out_lib_dir}/graph.tar -C ${out_lib_dir}
            COMMAND ${AR_EXEC} rcs ${GRAPH_LIB} ${out_lib_dir}/*.o
            DEPENDS ${tvm_py_script}
    )
    add_executable(${target_name} ${source_files} ${tvm_gens})
    target_link_libraries(${target_name} tvm-rt)
    target_link_directories(${target_name} PRIVATE ${out_lib_dir})
    # whole-archive ensure to force link the libgraph
    target_link_options(${target_name} PRIVATE -Wl,--whole-archive -lgraph -Wl,--no-whole-archive)
endfunction()


## resnet-50
set(resnet-50_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-50)
set(resnet-50_tvm_gen ${resnet-50_lib_dir}/graph.json.c)
set(resnet-50_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/resnet-50/scripts/resnet-50.py)
set(resnet-50_sources ${CMAKE_CURRENT_SOURCE_DIR}/resnet-50/resnet-50.c)
custom_tvm_executable(resnet-50.wasm ${resnet-50_sources} ${resnet-50_tvm_gen} ${resnet-50_tvm_script} ${resnet-50_lib_dir})

## vgg-19
set(vgg-19_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-19)
set(vgg-19_tvm_gen ${vgg-19_lib_dir}/graph.json.c)
set(vgg-19_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/vgg-19/scripts/vgg-19.py)
set(vgg-19_sources ${CMAKE_CURRENT_SOURCE_DIR}/vgg-19/vgg-19.c)
custom_tvm_executable(vgg-19.wasm ${vgg-19_sources} ${vgg-19_tvm_gen} ${vgg-19_tvm_script} ${vgg-19_lib_dir})

## mobilenet0.25
set(mobilenet0.25_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet0.25)
set(mobilenet0.25_tvm_gen ${mobilenet0.25_lib_dir}/graph.json.c ${mobilenet0.25_lib_dir}/graph.params.c)
set(mobilenet0.25_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/scripts/mobilenet0.25.py)
set(mobilenet0.25_sources ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/mobilenet0.25.c)
custom_tvm_executable(mobilenet0.25.wasm ${mobilenet0.25_sources} ${mobilenet0.25_tvm_gen} ${mobilenet0.25_tvm_script} ${mobilenet0.25_lib_dir})

## yolo_v2
set(yolo_v2_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/yolo_v2)
set(yolo_v2_tvm_gen ${yolo_v2_lib_dir}/graph.json.c)
set(yolo_v2_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/yolo_v2/scripts/yolo_v2.py)
set(yolo_v2_sources ${CMAKE_CURRENT_SOURCE_DIR}/yolo_v2/yolo_v2.c)
custom_tvm_executable(yolo_v2.wasm ${yolo_v2_sources} ${yolo_v2_tvm_gen} ${yolo_v2_tvm_script} ${yolo_v2_lib_dir})
