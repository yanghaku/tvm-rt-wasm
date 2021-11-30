/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/c_backend_api.c
 * \brief the implement for c_backend_api.h
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#include <stdio.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/utils/common.h>

/*!
 * \brief Backend function for modules to get function
 *  from its environment mod_node (its imports and global function).
 *  The user do should not call TVMFuncFree on func.
 *
 * \param mod_node The module handle.
 * \param func_name The name of the function.
 * \param out The result function.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendGetFuncFromEnv(void *mod_node, const char *func_name, TVMFunctionHandle *out) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Backend function to register system-wide library symbol.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendRegisterSystemLibSymbol(const char *name, void *ptr) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Backend function to allocate temporal workspace.
 *
 * \note The result allocated space is ensured to be aligned to kTempAllocaAlignment.
 *
 * \param nbytes The size of the space requested.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \param dtype_code_hint The type code of the array elements. Only used in
 * certain backends such as OpenGL.
 * \param dtype_bits_hint The type bits of the array elements. Only used in
 * certain backends such as OpenGL.
 * \return nullptr when error is thrown, a valid ptr if success
 */
TVM_DLL void *TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes, int dtype_code_hint,
                                       int dtype_bits_hint) {
    return 0;
}

/*!
 * \brief Backend function to free temporal workspace.
 *
 * \param ptr The result allocated space pointer.
 * \param device_type The device type which the space will be allocated.
 * \param device_id The device id which the space will be allocated.
 * \return 0 when no error is thrown, -1 when failure happens
 *
 * \sa TVMBackendAllocWorkspace
 */
TVM_DLL int TVMBackendFreeWorkspace(int device_type, int device_id, void *ptr) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Backend function to register execution environment(e.g. python)
 *        specific C APIs.
 *
 * \note  We only register the C API function when absolutely necessary (e.g. when signal handler
 *  cannot trap back into python). In most cases we should use the PackedFunc FFI.
 *
 * \param name The name of the symbol
 * \param ptr The symbol address.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendRegisterEnvCAPI(const char *name, void *ptr) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Backend function for running parallel jobs.
 *
 * \param flambda The parallel function to be launched.
 * \param cdata The closure data.
 * \param num_task Number of tasks to launch, can be 0, means launch
 *           with all available threads.
 *
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void *cdata, int num_task) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief BSP barrrier between parallel threads
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv *penv) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}

/*!
 * \brief Simple static initialization function.
 *  Run f once and set handle to be not null.
 *  This function is mainly used for test purpose.
 *
 * \param handle A global address to indicate f
 * \param f The function to be run
 * \param cdata The closure data to pass to the function.
 * \param nbytes Number of bytes in the closure data.
 * \return 0 when no error is thrown, -1 when failure happens
 */
TVM_DLL int TVMBackendRunOnce(void **handle, int (*f)(void *), void *cdata, int nbytes) {
    // todo: implement this api
    SET_ERROR_RETURN(-1, "This API has not yet been implemented");
}
