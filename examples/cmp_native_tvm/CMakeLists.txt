cmake_minimum_required(VERSION 3.19)
project(tvm-rt-wasm-cmp-native-tvm)

# tvm installed path
if (NOT DEFINED TVM_ROOT)
    set(TVM_ROOT /opt/tvm)
endif ()
message("native tvm path = ${TVM_ROOT}")
include_directories(. ${TVM_ROOT}/include)
link_directories(${TVM_ROOT}/lib)
link_libraries(tvm_runtime)

# dlpack path
if (NOT DEFINED DLPACK_PATH)
    set(DLPACK_PATH ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/dlpack)
endif ()
include_directories(${DLPACK_PATH}/include)

# cuda toolkit path
find_package(CUDAToolkit REQUIRED)
message("link cuda path = ${CUDAToolkit_LIBRARY_DIR}")
# build for native need link with cuda library
link_libraries(CUDA::cuda_driver)

# set the ar path
set(AR_EXEC llvm-ar)

set(TVM_HOST_TARGET "native")
set(TVM_DEVICE_TARGET cuda) # default device target for cuda
message("Native TVM build target: host=${TVM_HOST_TARGET}, device=${TVM_DEVICE_TARGET}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
                -w -Wextra -Werror -Wl,--gc-sections")
set(EXAMPLE_PATH "../c")
include_directories(${EXAMPLE_PATH}/common_headers)

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
    target_link_directories(${target_name} PRIVATE ${out_lib_dir})
    # whole-archive ensure to force link the libgraph
    target_link_options(${target_name} PRIVATE -Wl,--whole-archive -lgraph -Wl,--no-whole-archive)
endfunction()


## resnet-50
set(resnet-50_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-50)
set(resnet-50_tvm_gen ${resnet-50_lib_dir}/graph.json.c)
set(resnet-50_sources ${EXAMPLE_PATH}/resnet-50/resnet-50.c)
custom_tvm_executable(resnet-50 ${resnet-50_sources} ${resnet-50_tvm_gen} "resnet-50" ${resnet-50_lib_dir})

## vgg-16
set(vgg-16_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-16)
set(vgg-16_tvm_gen ${vgg-16_lib_dir}/graph.json.c)
set(vgg-16_sources ${EXAMPLE_PATH}/vgg-16/vgg-16.c)
custom_tvm_executable(vgg-16 ${vgg-16_sources} ${vgg-16_tvm_gen} "vgg-16" ${vgg-16_lib_dir})

## mobilenet
set(mobilenet_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet)
set(mobilenet_tvm_gen "${mobilenet_lib_dir}/graph.json.c")
set(mobilenet_sources ${EXAMPLE_PATH}/mobilenet/mobilenet.c)
custom_tvm_executable(mobilenet ${mobilenet_sources} ${mobilenet_tvm_gen} "mobilenet" ${mobilenet_lib_dir})

## bert-large-uncased
set(bert-large-uncased_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/bert-large-uncased)
set(bert-large-uncased_tvm_gen ${bert-large-uncased_lib_dir}/graph.json.c)
set(bert-large-uncased_sources ${EXAMPLE_PATH}/bert-large-uncased/bert-large-uncased.c)
custom_tvm_executable(bert-large-uncased ${bert-large-uncased_sources} ${bert-large-uncased_tvm_gen} "bert-large-uncased" ${bert-large-uncased_lib_dir})
