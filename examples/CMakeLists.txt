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
else ()
    link_directories(${CUDAToolkit_ROOT}/lib64)
    link_directories(${CUDAToolkit_ROOT}/lib64/stubs)
    link_directories(${CUDAToolkit_ROOT}/lib/)
    link_directories(${CUDAToolkit_ROOT}/lib/x64)
    set(PY_RUNTIME native)
endif ()

# whether use cuda
message("examples USE_CUDA = ${EXAMPLE_USE_CUDA}")
if (${EXAMPLE_USE_CUDA})
    message("examples CUDAToolkit path = ${CUDAToolkit_ROOT}")
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    if (${USE_WASI_SDK})
        add_link_options(-Wl,--allow-undefined)
    else ()
        # build for native need link with cuda library
        link_libraries(cuda)
    endif ()
    SET(PY_TARGET cuda)
else ()
    link_libraries(m)
    SET(PY_TARGET cpu)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-examples)

add_compile_options(-fno-exceptions -O3 -w -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-function -fvisibility-inlines-hidden -flto -Wl,--strip-all)

#include_directories(${TVM_RT_WASM_ROOT}/include)
#link_directories(${TVM_RT_WASM_ROOT}/lib)

find_package(Python3 COMPONENTS Interpreter)

# mobilenet0.25
set(mobilenet0.25_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet0.25)
set(mobilenet0.25.tvm_gen ${mobilenet0.25_lib_dir}/graph.json.c ${mobilenet0.25_lib_dir}/graph.params.c)
set(mobilenet0.25.tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/scripts/mobilenet0.25.py)
add_custom_command(
        OUTPUT ${mobilenet0.25.tvm_gen}
        COMMAND python ${mobilenet0.25.tvm_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${mobilenet0.25_lib_dir}
        COMMAND tar -xf ${mobilenet0.25_lib_dir}/graph.tar -C ${mobilenet0.25_lib_dir}
        COMMAND ${AR_EXEC} -r ${mobilenet0.25_lib_dir}/graph.lib ${mobilenet0.25_lib_dir}/*.o
        COMMAND ${AR_EXEC} -r ${mobilenet0.25_lib_dir}/libgraph.a ${mobilenet0.25_lib_dir}/*.o
        DEPENDS ${mobilenet0.25.tvm_script}
)
add_executable(mobilenet0.25.wasm mobilenet0.25/mobilenet0.25.c ${mobilenet0.25.tvm_gen})
target_link_libraries(mobilenet0.25.wasm tvm-rt)
target_link_directories(mobilenet0.25.wasm PRIVATE ${mobilenet0.25_lib_dir})
target_link_libraries(mobilenet0.25.wasm graph)

# resnet-50
set(resnet-50_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-50)
set(resnet-50_tvm_gen ${resnet-50_lib_dir}/graph.json.c)
set(resnet-50_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/resnet-50/scripts/resnet-50.py)
add_custom_command(
        OUTPUT ${resnet-50_tvm_gen}
        COMMAND python ${resnet-50_tvm_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${resnet-50_lib_dir}
        COMMAND tar -xf ${resnet-50_lib_dir}/graph.tar -C ${resnet-50_lib_dir}
        COMMAND ${AR_EXEC} -r ${resnet-50_lib_dir}/graph.lib ${resnet-50_lib_dir}/*.o
        COMMAND ${AR_EXEC} -r ${resnet-50_lib_dir}/libgraph.a ${resnet-50_lib_dir}/*.o
        DEPENDS ${resnet-50_tvm_script}
)
add_executable(resnet-50.wasm resnet-50/resnet-50.c ${resnet-50_tvm_gen})
target_link_libraries(resnet-50.wasm tvm-rt)
target_link_directories(resnet-50.wasm PRIVATE ${resnet-50_lib_dir})
target_link_libraries(resnet-50.wasm graph)

# vgg-19
set(vgg-19_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-19)
set(vgg-19_tvm_gen ${vgg-19_lib_dir}/graph.json.c)
set(vgg-19_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/vgg-19/scripts/vgg-19.py)
add_custom_command(
        OUTPUT ${vgg-19_tvm_gen}
        COMMAND python ${vgg-19_tvm_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${vgg-19_lib_dir}
        COMMAND tar -xf ${vgg-19_lib_dir}/graph.tar -C ${vgg-19_lib_dir}
        COMMAND ${AR_EXEC} -r ${vgg-19_lib_dir}/graph.lib ${vgg-19_lib_dir}/*.o
        COMMAND ${AR_EXEC} -r ${vgg-19_lib_dir}/libgraph.a ${vgg-19_lib_dir}/*.o
        DEPENDS ${vgg-19_tvm_script}
)
add_executable(vgg-19.wasm vgg-19/vgg-19.c ${vgg-19_tvm_gen})
target_link_libraries(vgg-19.wasm tvm-rt)
target_link_directories(vgg-19.wasm PRIVATE ${vgg-19_lib_dir})
target_link_libraries(vgg-19.wasm graph)

# yolo_v2
set(yolo_v2_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/yolo_v2)
set(yolo_v2_tvm_gen ${yolo_v2_lib_dir}/graph.json.c)
set(yolo_v2_tvm_script ${CMAKE_CURRENT_SOURCE_DIR}/yolo_v2/scripts/yolo_v2.py)
add_custom_command(
        OUTPUT ${yolo_v2_tvm_gen}
        COMMAND python ${yolo_v2_tvm_script} --runtime ${PY_RUNTIME} --target ${PY_TARGET} -o ${yolo_v2_lib_dir}
        COMMAND tar -xf ${yolo_v2_lib_dir}/graph.tar -C ${yolo_v2_lib_dir}
        COMMAND ${AR_EXEC} -r ${yolo_v2_lib_dir}/graph.lib ${yolo_v2_lib_dir}/*.o
        COMMAND ${AR_EXEC} -r ${yolo_v2_lib_dir}/libgraph.a ${yolo_v2_lib_dir}/*.o
        DEPENDS ${yolo_v2_tvm_script}
)
add_executable(yolo_v2.wasm yolo_v2/yolo_v2.c ${yolo_v2_tvm_gen})
target_link_libraries(yolo_v2.wasm tvm-rt)
target_link_directories(yolo_v2.wasm PRIVATE ${yolo_v2_lib_dir})
target_link_libraries(yolo_v2.wasm graph)
