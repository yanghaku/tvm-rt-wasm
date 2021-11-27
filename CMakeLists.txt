cmake_minimum_required(VERSION 3.19)

# the dependence prefix
set(WASI_SDK_PREFIX D:/lib/wasi-sdk)
set(CUDAToolkit_ROOT D:/software/cuda/toolkit)

# wasi-sdk
set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
project(tvm-rt)

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-fno-exceptions -O3")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fno-exceptions -O3")

include_directories(include)
include_directories(3rdparty/dlpack/include)
include_directories(${CUDAToolkit_ROOT}/include)

set(LIB_SRC)
file(GLOB_RECURSE LIB_SRC src/runtime/*.c)
add_library(tvm-rt STATIC ${LIB_SRC})


install(TARGETS tvm-rt ${CMAKE_INSTALL_LIBDIR})
set(INSTALL_INCLUDE_FILES)
file(GLOB INSTALL_INCLUDE_FILES include/tvm/runtime/*.h)
install(FILES ${INSTALL_INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/tvm/runtime/)
