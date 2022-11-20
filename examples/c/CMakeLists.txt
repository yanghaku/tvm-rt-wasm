cmake_minimum_required(VERSION 3.19)

# wasi-sdk
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()

# emsdk path
if (NOT DEFINED EMSDK_PREFIX)
    set(EMSDK_PREFIX /opt/emsdk)
endif ()

# cuda-wasm-toolkit path (wasm32-wasi + cuda)
if (NOT DEFINED CUDA_WASM_TOOLKIT_ROOT)
    set(CUDA_WASM_TOOLKIT_ROOT /opt/cuda-wasm-toolkit/)
endif ()

# tvm-rt-wasm installed path
if (NOT DEFINED TVM_RT_WASM_ROOT)
    set(TVM_RT_WASM_ROOT /opt/tvm-rt-wasm)
endif ()

# set the ar path
set(AR_EXEC llvm-ar)

option(USE_WASI_SDK "USE WASI SDK" OFF)
option(USE_EMSDK "USE emscripten toolchain" OFF)
option(EXAMPLE_USE_CUDA "example use cuda" ON)

if (${USE_EMSDK} AND ${USE_WASI_SDK})
    message(FATAL_ERROR "Cannot use wasi-sdk and emscripten at the same time!")
endif ()
if (${USE_EMSDK} AND ${EXAMPLE_USE_CUDA})
    message(FATAL_ERROR "Emscripten target cannot support using cuda !")
endif ()


message("examples USE WASI-SDK = ${USE_WASI_SDK}")
message("USE USE_EMSDK = ${USE_EMSDK}")
if (${USE_WASI_SDK})
    # setup wasi-sdk as toolchain
    message("examples WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    set(TVM_HOST_TARGET "wasm32-wasi")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mbulk-memory")
    add_link_options(-flto -Wl,--lto-O3,--gc-sections)
else ()
    # setup emsdk as toolchain
    if (${USE_EMSDK})
        message("example EMSDK path=${EMSDK_PREFIX}")
        set(CMAKE_TOOLCHAIN_FILE ${EMSDK_PREFIX}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake)
        set(TVM_HOST_TARGET "wasm32-emscripten")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mbulk-memory -sNODERAWFS=1 -sTOTAL_MEMORY=1689976832")
        add_link_options(-flto -Wl,--allow-undefined,--lto-O3,--gc-sections)
    else () # native toolchain
        link_libraries(m)
    endif ()
endif (${USE_WASI_SDK})


# whether use cuda
message("examples USE_CUDA = ${EXAMPLE_USE_CUDA}")
if (${EXAMPLE_USE_CUDA})
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    if (NOT ${USE_WASI_SDK})
        # cuda toolkit path
        find_package(CUDAToolkit REQUIRED)
        message("examples link cuda path = ${CUDAToolkit_LIBRARY_DIR}")
        # build for native need link with cuda library
        link_libraries(CUDA::cuda_driver)
    else ()
        # wasm32-wasi + cuda
        add_link_options(${CUDA_WASM_TOOLKIT_ROOT}/lib/cuda-driver-stub.bc -Wl,--allow-undefined-file=${CUDA_WASM_TOOLKIT_ROOT}/share/defined-symbols.txt)
    endif ()
    if (NOT DEFINED TVM_DEVICE_TARGET)
        set(TVM_DEVICE_TARGET cuda) # default device target for cuda
    endif ()
endif ()

if (NOT DEFINED TVM_DEVICE_TARGET)
    set(TVM_DEVICE_TARGET "cpu") # if not use cuda, default device target is cpu
endif ()
if (NOT DEFINED TVM_HOST_TARGET)
    set(TVM_HOST_TARGET "native")
endif ()
message("TVM build target: host=${TVM_HOST_TARGET}, device=${TVM_DEVICE_TARGET}")

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-c-examples)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
                -w -Wextra -Werror -Wl,--gc-sections")

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

    if (MSVC)
        set(GRAPH_LIB ${out_lib_dir}/graph.lib)
    else ()
        set(GRAPH_LIB ${out_lib_dir}/libgraph.a)
    endif ()
    add_custom_command(
            OUTPUT ${tvm_gens}
            COMMAND python ${tvm_build_script} --model ${model} --host-target ${TVM_HOST_TARGET} --device-target ${TVM_DEVICE_TARGET} -o ${out_lib_dir}
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
