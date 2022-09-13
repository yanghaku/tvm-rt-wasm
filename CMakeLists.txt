cmake_minimum_required(VERSION 3.19)

# the dependence prefix
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()
if (NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT /usr/local/cuda)
endif ()
OPTION(USE_WASI_SDK "USE WASI-SDK" ON)
OPTION(USE_CUDA "USE CUDA" ON)
OPTION(ENABLE_SHARED "Share GPU memory in serverless platform" ON)

# wasi-sdk
message("USE WASI-SDK = ${USE_WASI_SDK}")
if (${USE_WASI_SDK})
    message("WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
endif ()

message("USE_CUDA = ${USE_CUDA}")
if (${USE_CUDA})
    message("CUDAToolkit path = ${CUDAToolkit_ROOT}")
    add_definitions(-DUSE_CUDA=1)
    include_directories(${CUDAToolkit_ROOT}/include)
else ()
    add_definitions(-DUSE_CUDA=0)
endif ()

message("ENABLE_SHARED = ${ENABLE_SHARED}")
if (${USE_CUDA} AND ${ENABLE_SHARED})
    add_definitions(-DENABLE_SHARED=1)
    include_directories(3rdparty/wasmer)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
            -w -Wextra -Werror -Wno-error=unused-parameter -Wno-error=format-extra-args \
            -Wno-error=unused-function -fvisibility=hidden")

include_directories(include 3rdparty/dlpack/include)
add_definitions(-DTVM_EXPORTS)

# now many devices such as jetson nano cannot support cuda11.4+
OPTION(CUDA_10_ONLY "ONLY USE CUDA 10.2.89 API" ON)
if (${CUDA_10_ONLY})
    message("use cuda version 10.2.89 API")
    add_definitions(-DCUDA_10_ONLY)
endif ()


set(LIB_SRC)
file(GLOB_RECURSE LIB_SRC src/runtime/*.c)
add_library(tvm-rt STATIC ${LIB_SRC})


install(TARGETS tvm-rt ${CMAKE_INSTALL_LIBDIR})
# only contain three public header file
install(FILES include/tvm/runtime/c_runtime_api.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES include/tvm/runtime/graph_executor_manager.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlpack/)

add_subdirectory(examples EXCLUDE_FROM_ALL)
