cmake_minimum_required(VERSION 3.19)

# the dependence prefix
if (NOT DEFINED WASI_SDK_PREFIX)
    set(WASI_SDK_PREFIX /opt/wasi-sdk)
endif ()
if (NOT DEFINED EMSDK_PREFIX)
    set(EMSDK_PREFIX /opt/emsdk)
endif ()

option(USE_EMSDK "USE emscripten toolchain" OFF)
option(USE_WASI_SDK "USE wasi-sdk toolchain" OFF)
option(USE_CUDA "USE CUDA" ON)

if (${USE_EMSDK} AND ${USE_WASI_SDK})
    message(FATAL_ERROR "Cannot use wasi-sdk and emscripten at the same time!")
endif ()

# setup emsdk as toolchain
message("USE USE_EMSDK = ${USE_EMSDK}")
if (${USE_EMSDK})
    message("EMSDK path=${EMSDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${EMSDK_PREFIX}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
endif ()

# setup wasi-sdk as toolchain
message("USE WASI-SDK = ${USE_WASI_SDK}")
if (${USE_WASI_SDK})
    message("WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
endif ()

# cuda support
message("USE_CUDA = ${USE_CUDA}")
# now many devices such as jetson nano cannot support cuda11.4+, so use 10.2 as the default cuda version
option(CUDA_10_ONLY "ONLY USE CUDA 10.2.89 API" ON)
if (${USE_CUDA})
    add_definitions(-DUSE_CUDA=1)
    include_directories(3rdparty/cuda/include)
    if (${CUDA_10_ONLY})
        message("use cuda version 10.2.89 API")
        add_definitions(-DCUDA_10_ONLY)
    endif ()
else ()
    add_definitions(-DUSE_CUDA=0)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -fno-exceptions -O3 -ffunction-sections -fdata-sections -fno-rtti -fPIC \
            -w -Wextra -Werror -Wno-error=unused-parameter -Wno-error=format-extra-args \
            -Wno-error=unused-function -fvisibility=hidden")

include_directories(include 3rdparty/dlpack/include)
add_definitions(-DTVM_EXPORTS)


set(LIB_SRC)
file(GLOB_RECURSE LIB_SRC src/runtime/*.c)
add_library(tvm-rt STATIC ${LIB_SRC})


# set default install path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX /opt/tvm-rt-wasm/)
endif ()
install(TARGETS tvm-rt ${CMAKE_INSTALL_LIBDIR})
# only contain three public header file
install(FILES include/tvm/runtime/c_runtime_api.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES include/tvm/runtime/graph_executor_manager.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlpack/)

add_subdirectory(examples/c EXCLUDE_FROM_ALL)
