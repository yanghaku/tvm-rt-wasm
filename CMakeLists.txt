cmake_minimum_required(VERSION 3.19)

##---------------------- Project options and default variables ---------------------------------------------------------
## Project options:
## * USE_WASI_SDK: If use the wasi-sdk, set `/path/to/wasi-sdk` or set `ON`/`AUTO`/`TRUE` to use default `/opt/wasi-sdk`.
## * USE_EMSDK: If use the emscripten, set `/path/to/emsdk` or set `ON`/`AUTO`TRUE` to use default `/opt/emsdk`.

# Default build type: Release
# Default install path: /opt/tvm-rt-wasm/
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX /opt/tvm-rt-wasm/ CACHE PATH "" FORCE)
endif ()

# Used for the developers.
option(BUILD_EXAMPLES "Build the examples." OFF)
option(BUILD_TESTS "Build the tests." OFF)
option(BUILD_MEMORY_CHECK "Build with the memory sanitize." OFF)

#------------------------- Check options and setup toolchain -----------------------------------------------------------
set(_ENABLE_CUDA ON)
set(_ENABLE_WEBGPU ON)

macro(_is_bool_value _str _ret)
    string(TOUPPER ${_str} _s)
    if (${_s} STREQUAL "ON" OR ${_s} STREQUAL "TRUE" OR ${_s} STREQUAL "YES" OR ${_s} STREQUAL "AUTO" OR ${_s} STREQUAL "1")
        set(${_ret} ON)
    else ()
        set(${_ret} OFF)
    endif ()
endmacro()

if (DEFINED USE_WASI_SDK)
    message(STATUS "Use the wasi-sdk to build. Disable WebGPU support.")
    _is_bool_value("${USE_WASI_SDK}" _use_default)
    if (${_use_default})
        set(WASI_SDK_PREFIX /opt/wasi-sdk)
        message(STATUS "Use the default wasi-sdk path: ${WASI_SDK_PREFIX}")
    else ()
        set(WASI_SDK_PREFIX ${USE_WASI_SDK})
        message(STATUS "Use the wasi-sdk path: ${WASI_SDK_PREFIX}")
    endif ()

    set(_ENABLE_WEBGPU OFF)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(_ENABLE_WASI_SDK ON)
else ()
    set(_ENABLE_WASI_SDK OFF)
endif ()

if (DEFINED USE_EMSDK)
    message(STATUS "Use the emscripten to build. Disable CUDA support.")
    _is_bool_value("${USE_EMSDK}" _use_default)
    if (${_use_default})
        set(USE_EMSDK /opt/emsdk)
        message(STATUS "Use the default emscripten path = ${USE_EMSDK}")
    else ()
        message(STATUS "Use the emscripten path = ${USE_EMSDK}")
    endif ()

    set(_ENABLE_CUDA OFF)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mbulk-memory -fPIC")
    set(CMAKE_TOOLCHAIN_FILE ${USE_EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake)
    set(_ENABLE_EMSDK ON)
else ()
    set(_ENABLE_EMSDK OFF)
endif ()

if (${_ENABLE_WASI_SDK} AND ${_ENABLE_EMSDK})
    message(FATAL_ERROR "Cannot use wasi-sdk and emscripten at the same time!")
endif ()

#------------------------- Project -------------------------------------------------------------------------------------
project(tvm-rt-wasm LANGUAGES C VERSION 0.2.0)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_definitions(-DTVM_EXPORTS)
include_directories(include 3rdparty/dlpack/include 3rdparty/tvm/include src/core)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fno-exceptions -ffunction-sections -fdata-sections -fvisibility=hidden")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wno-error=unused-function")

## Core library target
file(GLOB_RECURSE _LIB_CORE_SRC src/core/*.c)
add_library(tvm-rt-core STATIC ${_LIB_CORE_SRC})
target_include_directories(tvm-rt-core INTERFACE $<INSTALL_INTERFACE:include>)

## Backend library targets.
function(add_backend_library _target_name _backend_folder)
    file(GLOB_RECURSE _backend_sources ${_backend_folder}/*.c)
    add_library(${_target_name} STATIC ${_backend_sources})
    target_include_directories(${_target_name} PRIVATE src/backends)
endfunction()
add_backend_library(tvm-rt-backend-graph src/backends/graph)
add_backend_library(tvm-rt-backend-aot src/backends/aot)
add_backend_library(tvm-rt-backend-relay-vm src/backends/relay_vm)

## CUDA library target.
if (${_ENABLE_CUDA})
    # Now many devices such as jetson nano cannot support cuda11.4+, so use 10.2 as the default cuda version.
    option(CUDA_10_ONLY "Only use CUDA 10.2.89 API. (No CUDA graph support.)" ON)
    if (${CUDA_10_ONLY})
        add_definitions(-DCUDA_10_ONLY)
    endif ()

    file(GLOB_RECURSE _LIB_CUDA_SRC src/accelerators/cuda/*.c)
    add_library(tvm-rt-cuda STATIC ${_LIB_CUDA_SRC})
    target_include_directories(tvm-rt-cuda PRIVATE 3rdparty/cuda/include src/accelerators/cuda src/backends)
endif ()

## WebGPU library target.
if (${_ENABLE_WEBGPU})
    file(GLOB_RECURSE _LIB_WEBGPU_SRC src/accelerators/webgpu/*.c)
    add_library(tvm-rt-webgpu STATIC ${_LIB_WEBGPU_SRC})
    target_include_directories(tvm-rt-webgpu PRIVATE src/accelerators/webgpu)
    if (NOT ${_ENABLE_EMSDK})
        target_include_directories(tvm-rt-webgpu PRIVATE 3rdparty/webgpu-headers)
    endif ()
endif ()

# js module
if (${_ENABLE_EMSDK})
    set(_empty_source ${CMAKE_CURRENT_BINARY_DIR}/cmake/null.c)
    file(WRITE ${_empty_source} "")
    add_executable(tvm-rt-module EXCLUDE_FROM_ALL ${_empty_source})
    target_link_options(tvm-rt-module PRIVATE --no-entry -O3 -flto -fno-exceptions -fno-rtti -Wl,--gc-sections
            -sMAIN_MODULE=2 -sWASM_BIGINT=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB
            -sERROR_ON_UNDEFINED_SYMBOLS=1 -sMODULARIZE=1 -sSINGLE_FILE=1 -sEXPORT_NAME="tvm_rt_module_create"
            -sEXPORTED_RUNTIME_METHODS=["FS","stackAlloc","stackSave","stackRestore","stringToUTF8","UTF8ToString","stringToNewUTF8","stringToUTF8OnStack"])

    set(_tvm_module_lib_deps tvm-rt-backend-graph tvm-rt-backend-aot tvm-rt-backend-relay-vm tvm-rt-core)
    if (${_ENABLE_WEBGPU})
        target_link_options(tvm-rt-module PRIVATE -sASYNCIFY)
        list(APPEND _tvm_module_lib_deps -Wl,--whole-archive tvm-rt-webgpu -Wl,--no-whole-archive)
    endif ()
    target_link_libraries(tvm-rt-module PRIVATE ${_tvm_module_lib_deps})
endif ()

# install targets
set(CMAKE_INSTALL_CMAKE_DIR lib/cmake/tvm-rt-wasm)

function(install_library _target_name)
    install(TARGETS ${_target_name} EXPORT tvm-rt-wasm-targets)
endfunction()
install_library(tvm-rt-core)
install_library(tvm-rt-backend-graph)
install_library(tvm-rt-backend-aot)
install_library(tvm-rt-backend-relay-vm)
if (${_ENABLE_CUDA})
    install_library(tvm-rt-cuda)
endif ()
if (${_ENABLE_WEBGPU})
    install_library(tvm-rt-webgpu)
endif ()

# install targets.cmake
install(EXPORT tvm-rt-wasm-targets FILE tvm-rt-wasm-targets.cmake
        CONFIGURATIONS Release
        DESTINATION ${CMAKE_INSTALL_CMAKE_DIR}
        NAMESPACE tvm-rt-wasm::)
# install config-version.cmake and config.cmake
include(CMakePackageConfigHelpers)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config.cmake
        "include(\${CMAKE_CURRENT_LIST_DIR}/tvm-rt-wasm-targets.cmake)\n")
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config-version.cmake
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/tvm-rt-wasm-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_CMAKE_DIR})

# install public header files
install(FILES 3rdparty/tvm/include/tvm/runtime/c_runtime_api.h DESTINATION include/tvm/runtime/)
install(FILES 3rdparty/dlpack/include/dlpack/dlpack.h DESTINATION include/dlpack/)
install(FILES include/graph_executor.h
        include/aot_executor.h
        include/relay_vm.h
        DESTINATION include)

# install LICENSE file
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_PREFIX})

if (${BUILD_EXAMPLES})
    add_subdirectory(examples/c EXCLUDE_FROM_ALL)
endif ()

if (${BUILD_MEMORY_CHECK})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif ()
