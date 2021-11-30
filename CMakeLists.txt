cmake_minimum_required(VERSION 3.19)

# the dependence prefix
set(WASI_SDK_PREFIX D:/lib/wasi-sdk)
set(CUDAToolkit_ROOT D:/software/cuda/toolkit)

# wasi-sdk
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
project(tvm-rt)

add_compile_options(-fno-exceptions -O3 -Wall -Wextra -Werror -Wno-error=unused-parameter -Wno-error=unused-function -fvisibility-inlines-hidden -fno-rtti)

OPTION(USE_CUDA "USE CUDA" ON)
message("lib USE_CUDA=${USE_CUDA}")
if (${USE_CUDA})
    add_definitions(-DUSE_CUDA=1)
    add_link_options(-Wl,--allow-undefined)
    include_directories(${CUDAToolkit_ROOT}/include)
else ()
    add_definitions(-DUSE_CUDA=0)
endif ()

include_directories(include)
include_directories(3rdparty/dlpack/include)

set(LIB_SRC)
file(GLOB_RECURSE LIB_SRC src/runtime/*.c)
add_library(tvm-rt STATIC ${LIB_SRC})


install(TARGETS tvm-rt ${CMAKE_INSTALL_LIBDIR})
set(INSTALL_INCLUDE_FILES)
file(GLOB INSTALL_INCLUDE_FILES include/tvm/runtime/*.h)
install(FILES ${INSTALL_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/dlpack/)

add_subdirectory(examples EXCLUDE_FROM_ALL)
