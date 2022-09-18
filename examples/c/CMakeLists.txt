cmake_minimum_required(VERSION 3.19)

# wasi-sdk
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()

# cuda toolkit path
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT /usr/local/cuda)
endif ()

# tvm-rt-wasm installed path
if (NOT DEFINED TVM_RT_WASM_ROOT)
    set(TVM_RT_WASM_ROOT /opt/tvm-rt-wasm)
endif ()

# set the ar path
set(AR_EXEC llvm-ar)

option(USE_WASI_SDK "USE WASI SDK" OFF)
option(EXAMPLE_USE_CUDA "example use cuda" ON)
# target device
set(TARGET_DEV "cpu" STRING)

message("examples USE WASI-SDK = ${USE_WASI_SDK}")
if (${USE_WASI_SDK})
    message("examples WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(PY_RUNTIME wasm)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
    add_link_options(-flto -Wl,--allow-undefined,--lto-O3,--gc-sections,--strip-all)
else ()
    set(PY_RUNTIME native)
    link_libraries(m)

    # check if the linker is lld
    execute_process(COMMAND ${CMAKE_LINKER} --version OUTPUT_VARIABLE LD_VERSION_OUTPUT)
    if (${LD_VERSION_OUTPUT} MATCHES "LLD*")
        message("USE gold to replace lld")
        # lld cannot process .ctos to .init-array
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
        link_directories(${CUDAToolkit_ROOT}/lib64 ${CUDAToolkit_ROOT}/lib/)
        link_libraries(cuda)
    endif ()
    set(TARGET_DEV cuda)
else ()
    set(TARGET_DEV cpu)
endif ()
message("build target device = ${TARGET_DEV}")

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-c-examples)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
                -w -Wextra -Werror -Wl,--strip-all,--gc-sections")

include_directories(common_headers)

# if this CMakeLists.txt is in source top level, must include and link with the tvm-rt-wasm library
# if this CmakeLists.txt is built as a subdirectory in tvm-rt-wasm project, do not need this
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include_directories(${TVM_RT_WASM_ROOT}/include)
    link_directories(${TVM_RT_WASM_ROOT}/lib)
endif ()

find_package(Python3 COMPONENTS Interpreter)


function(custom_tvm_executable target_name source_files tvm_gens model out_lib_dir)
    set(tvm_build_script_dir ${CMAKE_CURRENT_SOURCE_DIR}/../tvm_gen_scripts)
    set(tvm_build_script ${tvm_build_script_dir}/main.py)

    if (${USE_WASI_SDK})
        set(target_name "${target_name}.wasm")
    endif ()

    if (MSVC)
        set(GRAPH_LIB ${out_lib_dir}/graph.lib)
    else ()
        set(GRAPH_LIB ${out_lib_dir}/libgraph.a)
    endif ()
    add_custom_command(
            OUTPUT ${tvm_gens}
            COMMAND python ${tvm_build_script} --model ${model} --runtime ${PY_RUNTIME} --target ${TARGET_DEV} -o ${out_lib_dir}
            COMMAND tar -xf ${out_lib_dir}/graph.tar -C ${out_lib_dir}
            COMMAND ${AR_EXEC} rcs ${GRAPH_LIB} ${out_lib_dir}/*.o
            DEPENDS ${tvm_build_script}
            WORKING_DIRECTORY ${tvm_build_script_dir}
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
set(resnet-50_sources resnet-50/resnet-50.c)
custom_tvm_executable(resnet-50 ${resnet-50_sources} ${resnet-50_tvm_gen} "resnet-50" ${resnet-50_lib_dir})

## vgg-16
set(vgg-16_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-16)
set(vgg-16_tvm_gen ${vgg-16_lib_dir}/graph.json.c)
set(vgg-16_sources vgg-16/vgg-16.c)
custom_tvm_executable(vgg-16 ${vgg-16_sources} ${vgg-16_tvm_gen} "vgg-16" ${vgg-16_lib_dir})

## mobilenet
set(mobilenet_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet)
set(mobilenet_tvm_gen "${mobilenet_lib_dir}/graph.json.c")
set(mobilenet_sources mobilenet/mobilenet.c)
custom_tvm_executable(mobilenet ${mobilenet_sources} ${mobilenet_tvm_gen} "mobilenet" ${mobilenet_lib_dir})

## yolo-v4
set(yolo-v4_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/yolo-v4)
set(yolo-v4_tvm_gen ${yolo-v4_lib_dir}/graph.json.c)
set(yolo-v4_sources yolo-v4/yolo-v4.c)
custom_tvm_executable(yolo-v4 ${yolo-v4_sources} ${yolo-v4_tvm_gen} "yolo-v4" ${yolo-v4_lib_dir})
