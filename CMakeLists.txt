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


# wasi-sdk
message("USE WASI-SDK = ${USE_WASI_SDK}")
if (${USE_WASI_SDK})
    message("WASI-SDK path=${WASI_SDK_PREFIX}")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
endif ()

message("USE_CUDA = ${USE_CUDA}")
if (${USE_CUDA})
    message("CUDAToolkit path = ${CUDAToolkit_ROOT}")
    add_definitions(-DUSE_CUDA=1)
    include_directories(${CUDAToolkit_ROOT}/include)
    if (${USE_WASI_SDK})
        add_link_options(-Wl,--allow-undefined)
    endif ()
else ()
    add_definitions(-DUSE_CUDA=0)
endif ()

#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm)

add_compile_options(-fno-exceptions -O3 -w -Wextra -Werror -Wno-error=unused-parameter -Wno-error=format-extra-args
        -Wno-error=unused-function -fvisibility=hidden -Wl,--strip-all)
SET(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -std=c11)

include_directories(include)
include_directories(3rdparty/dlpack/include)
add_definitions(-DTVM_EXPORTS)

# now many devices such as jetson nano cannot support cuda11.4+
add_definitions(-DCUDA_10_ONLY)

set(LIB_SRC)
file(GLOB_RECURSE LIB_SRC src/runtime/*.c)
add_library(tvm-rt STATIC ${LIB_SRC})


install(TARGETS tvm-rt ${CMAKE_INSTALL_LIBDIR})
# only contain three public header file
install(FILES include/tvm/runtime/c_runtime_api.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES include/tvm/runtime/graph_executor_manager.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlpack/)

add_subdirectory(examples EXCLUDE_FROM_ALL)
