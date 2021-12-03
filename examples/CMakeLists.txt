cmake_minimum_required(VERSION 3.19)

# config for examples

# wasi-sdk
set(WASI_SDK_PREFIX D:/lib/wasi-sdk)

# if use cuda
set(USE_CUDA OFF)

# tvm-rt-wasm installed path
set(TVM_RT_ROOT D:/lib/tvm-rt/)

#-----------------------------------------------------------------------------------------------------------------------

# wasi-sdk
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)

project(tvm-rt-examples)
message("examples for tvm-rt")

add_compile_options(-fno-exceptions -O3 -Wall -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-function -fvisibility-inlines-hidden -fno-rtti)
add_link_options(-Wl,--allow-undefined)

message("examples USE_CUDA = ${USE_CUDA}")

include_directories(${TVM_RT_ROOT}/include)
link_directories(${TVM_RT_ROOT}/lib)

if (${USE_CUDA})
    add_definitions(-DUSE_CUDA=1)
    SET(PY_TARGET cuda)
else ()
    SET(PY_TARGET cpu)
endif ()

find_package(Python3 COMPONENTS Interpreter)

# mobilenet0.25
execute_process(COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/scripts/mobilenet0.25.py --target ${PY_TARGET} -o ${CMAKE_CURRENT_SOURCE_DIR}/mobilenet0.25/lib/)
file(GLOB mobilenet0.25_LIB_C mobilenet0.25/lib/*.c)
file(GLOB mobilenet0.25_LIB_OBJ mobilenet0.25/lib/*.o)
add_executable(mobilenet0.25.wasm mobilenet0.25/mobilenet0.25.c ${mobilenet0.25_LIB_C} ${mobilenet0.25_LIB_OBJ})
target_link_libraries(mobilenet0.25.wasm tvm-rt)
