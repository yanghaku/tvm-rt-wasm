cmake_minimum_required(VERSION 3.19)

#------------------------- Setup toolchain -----------------------------------------------------------------------------
## * USE_WASI_SDK: If use the wasi-sdk, set `/path/to/wasi-sdk` or set `ON`/`AUTO`/`TRUE` to use default `/opt/wasi-sdk`.
## * USE_EMSDK: If use the emscripten, set `/path/to/emsdk` or set `ON`/`AUTO`TRUE` to use default `/opt/emsdk`.

macro(_is_bool_value _str _ret)
    string(TOUPPER ${_str} _s)
    if (${_s} STREQUAL "ON" OR ${_s} STREQUAL "TRUE" OR ${_s} STREQUAL "YES" OR ${_s} STREQUAL "AUTO" OR ${_s} STREQUAL "1")
        set(${_ret} ON)
    else ()
        set(${_ret} OFF)
    endif ()
endmacro()

if (DEFINED USE_WASI_SDK)
    message(STATUS "Use the wasi-sdk to build examples.")
    _is_bool_value("${USE_WASI_SDK}" _use_default)
    if (${_use_default})
        set(WASI_SDK_PREFIX /opt/wasi-sdk)
    else ()
        set(WASI_SDK_PREFIX ${USE_WASI_SDK})
    endif ()

    set(CMAKE_TOOLCHAIN_FILE ${WASI_SDK_PREFIX}/share/cmake/wasi-sdk.cmake)
    set(_ENABLE_WASI_SDK ON)
else ()
    set(_ENABLE_WASI_SDK OFF)
endif ()

if (DEFINED USE_EMSDK)
    message(STATUS "Use the emscripten to build examples.")
    _is_bool_value("${USE_EMSDK}" _use_default)
    if (${_use_default})
        set(USE_EMSDK /opt/emsdk)
    endif ()

    set(CMAKE_TOOLCHAIN_FILE ${USE_EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake)
    set(_ENABLE_EMSDK ON)
else ()
    set(_ENABLE_EMSDK OFF)
endif ()

# Check the options
if (${_ENABLE_EMSDK} AND ${_ENABLE_WASI_SDK})
    message(FATAL_ERROR "Cannot use wasi-sdk and emscripten at the same time!")
endif ()

# Example build devices, default is CPU.
option(EXAMPLE_USE_CUDA "Build the examples with WebGPU target." OFF)
option(EXAMPLE_USE_WEBGPU "Build the examples with CUDA target." OFF)

# Set the ar path
if (NOT DEFINED CMAKE_AR)
    set(CMAKE_AR llvm-ar)
endif ()

# Set python executable
if (NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python)
endif ()

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
#-----------------------------------------------------------------------------------------------------------------------
project(tvm-rt-wasm-c-examples LANGUAGES C)

if (${BUILD_MEMORY_CHECK})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
endif ()
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR) # This CMakeLists.txt is in source top level, must find tvm-rt-wasm package
    find_package(tvm-rt-wasm REQUIRED PATHS ${TVM_RT_WASM_ROOT})
else () # This CmakeLists.txt is built as a subdirectory in tvm-rt-wasm project, (for developers).
    function(_add_imported_library _target)
        add_library(tvm-rt-wasm::${_target} ALIAS ${_target})
    endfunction()

    if (${_ENABLE_CUDA})
        _add_imported_library(tvm-rt-cuda)
    endif ()
    if (${_ENABLE_WEBGPU})
        _add_imported_library(tvm-rt-webgpu)
    endif ()
    _add_imported_library(tvm-rt-backend-graph)
    _add_imported_library(tvm-rt-backend-aot)
    _add_imported_library(tvm-rt-backend-relay-vm)
    _add_imported_library(tvm-rt-backend-relax-vm)
    _add_imported_library(tvm-rt-core)
endif ()

include_directories(common_headers .)
# Setup compile and link flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fno-exceptions -ffunction-sections -fdata-sections -Wall -Wextra -Werror")
add_link_options(-Wl,--gc-sections -flto)
if (${_ENABLE_WASI_SDK})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msimd128 -mbulk-memory")
    set(CMAKE_EXECUTABLE_SUFFIX ".wasm")
    set(TVM_HOST_TARGET "wasm32-wasi")
elseif (${_ENABLE_EMSDK})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mbulk-memory")
    add_link_options(-Wl,--lto-O3 -sWASM_BIGINT=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sERROR_ON_UNDEFINED_SYMBOLS=1 -sNODERAWFS=1 -sSINGLE_FILE=1)
    set(TVM_HOST_TARGET "wasm32-emscripten")
else () # native toolchain
    link_libraries(m)
    set(TVM_HOST_TARGET "native")
    #    add_link_options(-Wl,--strip-all)
endif ()

# Setup CUDA library
message(STATUS "EXAMPLE_USE_CUDA = ${EXAMPLE_USE_CUDA}")
if (${EXAMPLE_USE_CUDA})
    add_definitions(-DEXAMPLE_USE_CUDA=1)
    link_libraries(-Wl,--whole-archive tvm-rt-wasm::tvm-rt-cuda -Wl,--no-whole-archive)
    if (NOT DEFINED TVM_DEVICE_TARGET)
        set(TVM_DEVICE_TARGET cuda) # default device target for cuda
    endif ()

    if (${_ENABLE_WASI_SDK})
        add_link_options(-Wl,--allow-undefined)
    elseif (${_ENABLE_EMSDK})
        message(FATAL_ERROR "Emscripten cannot support CUDA!")
    else ()
        # build for native, need to link with CUDA library
        find_package(CUDAToolkit REQUIRED)
        link_libraries(CUDA::cuda_driver)
    endif ()
endif ()

# Setup WebGPU library
message(STATUS "EXAMPLE_USE_WEBGPU = ${EXAMPLE_USE_WEBGPU}")
if (${EXAMPLE_USE_WEBGPU})
    add_definitions(-DEXAMPLE_USE_WEBGPU=1)
    link_libraries(-Wl,--whole-archive tvm-rt-wasm::tvm-rt-webgpu -Wl,--no-whole-archive)
    if (NOT DEFINED TVM_DEVICE_TARGET)
        set(TVM_DEVICE_TARGET webgpu) # default device target for WebGPU
    endif ()

    if (${_ENABLE_WASI_SDK})
        message(FATAL_ERROR "Wasi-sdk cannot support WebGPU now!")
    elseif (${_ENABLE_EMSDK})
        # Emscripten + WebGPU, run in nodejs. (the C examples cannot run in browser)
        add_link_options(-sASYNCIFY)
    else ()
        # native WebGPU, should link native WebGPU library, such dawn
        if (NOT DEFINED WEBGPU_NATIVE_LIB)
            message(FATAL_ERROR "Must set WEBGPU_NATIVE_LIB to link with native WebGPU library!")
        else ()
            link_libraries(${WEBGPU_NATIVE_LIB})
            link_libraries(dl pthread)
            # must use lld to link the libwgpu_native or libdawn
            add_link_options("-fuse-ld=lld")
        endif ()
    endif ()
endif ()

if (NOT DEFINED TVM_DEVICE_TARGET)
    set(TVM_DEVICE_TARGET "cpu") # if not use cuda, default device target is cpu
endif ()
message(STATUS "TVM build target: host=${TVM_HOST_TARGET}, device=${TVM_DEVICE_TARGET}.")


# executor: "graph", "aot", "relay_vm", "relax_vm"
function(add_tvm_bundle_executable target_name source_files model out_lib_dir executor)
    set(tvm_build_script_dir ${CMAKE_CURRENT_SOURCE_DIR}/../tvm_gen_scripts)
    set(tvm_build_script ${tvm_build_script_dir}/main.py)

    set(TVM_GEN_LIB ${out_lib_dir}/lib${executor}.a)
    if (${executor} STREQUAL "graph")
        set(GRAPH_JSON_OUTPUT "${out_lib_dir}/graph.json.c")
        list(APPEND source_files ${GRAPH_JSON_OUTPUT})
    elseif (${executor} STREQUAL "aot" OR ${executor} STREQUAL "relay_vm" OR ${executor} STREQUAL "relax_vm")
        set(GRAPH_JSON_OUTPUT "")
    else ()
        message(FATAL_ERROR "Unsupported executor ${executor}")
    endif ()
    add_custom_command(
            OUTPUT ${TVM_GEN_LIB} ${GRAPH_JSON_OUTPUT}
            COMMAND ${PYTHON_EXECUTABLE} ${tvm_build_script} --model ${model} -o ${out_lib_dir}
            --host-target ${TVM_HOST_TARGET} --device-target ${TVM_DEVICE_TARGET} --executor ${executor}
            COMMAND tar -xf ${out_lib_dir}/${executor}.tar -C ${out_lib_dir}
            COMMAND ${CMAKE_AR} rcs ${TVM_GEN_LIB} ${out_lib_dir}/*.o
            DEPENDS ${tvm_build_script}
            WORKING_DIRECTORY ${tvm_build_script_dir}
    )
    add_executable(${target_name} EXCLUDE_FROM_ALL ${source_files})
    add_custom_target("${target_name}_tvm_gen_lib" SOURCES ${TVM_GEN_LIB})
    add_dependencies(${target_name} "${target_name}_tvm_gen_lib")

    if (${executor} STREQUAL "relax_vm")
        # need whole archive or set undefined symbol
        target_link_libraries(${target_name} PRIVATE -Wl,--undefined=TVM_RT_WASM_RelaxExecutableModuleCreate
                tvm-rt-wasm::tvm-rt-backend-relax-vm tvm-rt-wasm::tvm-rt-core)
    else ()
        string(REPLACE "_" "-" backend_name ${executor})    # '_' replace to '-'
        target_link_libraries(${target_name} PRIVATE tvm-rt-wasm::tvm-rt-backend-${backend_name} tvm-rt-wasm::tvm-rt-core)
    endif ()

    string(REPLACE "-" "_" target_define ${target_name})    # '-' replace to '_'
    target_compile_definitions(${target_name} PUBLIC -D${target_define})
    # whole-archive ensure to force link the tvm generated lib.
    target_link_libraries(${target_name} PRIVATE -Wl,--whole-archive ${TVM_GEN_LIB} -Wl,--no-whole-archive)
endfunction()


## resnet-50
set(resnet-50_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-50)
set(resnet-50_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-50 ${resnet-50_sources} "resnet-50" ${resnet-50_lib_dir} "graph")

## vgg-16
set(vgg-16_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-16)
set(vgg-16_sources vgg/vgg.c)
add_tvm_bundle_executable(vgg-16 ${vgg-16_sources} "vgg-16" ${vgg-16_lib_dir} "graph")

## mobilenet
set(mobilenet_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet)
set(mobilenet_sources mobilenet/mobilenet.c)
add_tvm_bundle_executable(mobilenet ${mobilenet_sources} "mobilenet" ${mobilenet_lib_dir} "graph")

## bert-large-uncased
set(bert-large-uncased_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/bert-large-uncased)
set(bert-large-uncased_sources bert-uncased/bert-uncased.c)
add_tvm_bundle_executable(bert-large-uncased ${bert-large-uncased_sources} "bert-large-uncased" ${bert-large-uncased_lib_dir} "graph")

## resnet-18
set(resnet-18_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-18)
set(resnet-18_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-18 ${resnet-18_sources} "resnet-18" ${resnet-18_lib_dir} "graph")

## resnet-34
set(resnet-34_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-34)
set(resnet-34_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-34 ${resnet-34_sources} "resnet-34" ${resnet-34_lib_dir} "graph")

## resnet-101
set(resnet-101_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-101)
set(resnet-101_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-101 ${resnet-101_sources} "resnet-101" ${resnet-101_lib_dir} "graph")

## resnet-152
set(resnet-152_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-152)
set(resnet-152_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-152 ${resnet-152_sources} "resnet-152" ${resnet-152_lib_dir} "graph")

## vgg-19
set(vgg-19_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-19)
set(vgg-19_sources vgg/vgg.c)
add_tvm_bundle_executable(vgg-19 ${vgg-19_sources} "vgg-19" ${vgg-19_lib_dir} "graph")

## bert-base-uncased
set(bert-base-uncased_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/bert-base-uncased)
set(bert-base-uncased_sources bert-uncased/bert-uncased.c)
add_tvm_bundle_executable(bert-base-uncased ${bert-base-uncased_sources} "bert-base-uncased" ${bert-base-uncased_lib_dir} "graph")


## resnet-50-int8
set(resnet-50-int8_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/resnet-50-int8)
set(resnet-50-int8_sources resnet/resnet.c)
add_tvm_bundle_executable(resnet-50-int8 ${resnet-50-int8_sources} "resnet-50-int8" ${resnet-50-int8_lib_dir} "graph")

## vgg-16-int8
set(vgg-16-int8_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/vgg-16-int8)
set(vgg-16-int8_sources vgg/vgg.c)
add_tvm_bundle_executable(vgg-16-int8 ${vgg-16-int8_sources} "vgg-16-int8" ${vgg-16-int8_lib_dir} "graph")

## add mobilenet aot target
set(mobilenet-aot_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet-aot)
set(mobilenet-aot_sources mobilenet-aot/mobilenet-aot.c)
add_tvm_bundle_executable(mobilenet-aot ${mobilenet-aot_sources} "mobilenet" ${mobilenet-aot_lib_dir} "aot")

## add mobilenet relay vm target
## now the tvm cannot generate relay vm system-lib. todo: support it.
set(mobilenet-relay-vm_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet-relay-vm)
set(mobilenet-relay-vm_sources mobilenet-relay-vm/mobilenet-relay-vm.c)
add_tvm_bundle_executable(mobilenet-relay-vm ${mobilenet-relay-vm_sources} "mobilenet" ${mobilenet-relay-vm_lib_dir} "relay_vm")

## add mobilenet relax vm target
set(mobilenet-relax-vm_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/lib/mobilenet-relax-vm)
set(mobilenet-relax-vm_sources mobilenet-relax-vm/mobilenet-relax-vm.c)
add_tvm_bundle_executable(mobilenet-relax-vm ${mobilenet-relax-vm_sources} "mobilenet" ${mobilenet-relax-vm_lib_dir} "relax_vm")

## dso library test
if (NOT ${_ENABLE_WASI_SDK})
    function(add_dso_executable target_name sources)
        add_executable(${target_name} EXCLUDE_FROM_ALL ${sources})
        # link with all backends
        target_link_libraries(${target_name} PRIVATE
                tvm-rt-wasm::tvm-rt-backend-aot
                tvm-rt-wasm::tvm-rt-backend-graph
                tvm-rt-wasm::tvm-rt-backend-relay-vm
                -Wl,--whole-archive tvm-rt-wasm::tvm-rt-backend-relax-vm -Wl,--no-whole-archive
                tvm-rt-wasm::tvm-rt-core)
        target_compile_definitions(${target_name} PRIVATE -DDSO_TEST)
        if (${_ENABLE_EMSDK})
            target_compile_options(${target_name} PRIVATE -fPIC)
            target_link_options(${target_name} PRIVATE -fPIC -sMAIN_MODULE=2)
        endif ()
    endfunction()

    # for graph executor
    add_dso_executable(mobilenet-dso mobilenet/mobilenet.c)
    # for aot executor
    add_dso_executable(mobilenet-aot-dso mobilenet-aot/mobilenet-aot.c)
    # for relay vm
    add_dso_executable(mobilenet-relay-vm-dso mobilenet-relay-vm/mobilenet-relay-vm.c)
    # for relax vm
    add_dso_executable(mobilenet-relax-vm-dso mobilenet-relax-vm/mobilenet-relax-vm.c)
endif ()
