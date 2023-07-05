/**
 * @file device/device_api.h
 * @brief Define the device api base interface and member.
 */

#ifndef TVM_RT_WASM_CORE_DEVICE_DEVICE_API_H_INCLUDE_
#define TVM_RT_WASM_CORE_DEVICE_DEVICE_API_H_INCLUDE_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>

typedef struct DeviceAPI DeviceAPI;

/*------------------------The interface in device api---------------------------------------------*/

#define DEVICE_API_INTERFACE                                                                       \
                                                                                                   \
    /**                                                                                            \
     * @brief Set the environment device id to device.                                             \
     * @param dev_id The device_id to perform operation.                                           \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*SetDevice)(int dev_id);                                                                  \
                                                                                                   \
    /**                                                                                            \
     * @brief Allocate a data space on device.                                                     \
     * @param dev_id The device id to perform operation.                                           \
     * @param nbytes The number of bytes in memory.                                                \
     * @param alignment The alignment of the memory.                                               \
     * @param type_hint The type of elements.                                                      \
     * @return The allocated device pointer.  return NULL if fail.                                 \
     */                                                                                            \
    void *(*AllocDataSpace)(int dev_id, size_t nbytes, size_t alignment, DLDataType type_hint);    \
                                                                                                   \
    /**                                                                                            \
     * @brief Allocate a data space on device with memory scope support.                           \
     * @param dev_id The device id to perform operation.                                           \
     * @param ndim The number of dimension of allocated tensor.                                    \
     * @param shape The shape of allocated tensor.                                                 \
     * @param dtype The data type of elements.                                                     \
     * @param mem_scope The memory scope of allocated tensor.                                      \
     * @return The allocated device pointer.  return NULL if fail.                                 \
     */                                                                                            \
    void *(*AllocDataSpaceScope)(int dev_id, int ndim, const int64_t *shape, DLDataType dtype,     \
                                 const char *mem_scope);                                           \
                                                                                                   \
    /**                                                                                            \
     * @brief Free a data space on device.                                                         \
     * @param dev_id The device_id to perform operation.                                           \
     * @param ptr The data space.                                                                  \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*FreeDataSpace)(int dev_id, void *ptr);                                                   \
                                                                                                   \
    /**                                                                                            \
     * @brief Copy data from one DLTensor to DLTensor.                                             \
     * @param from The source DLTensor.                                                            \
     * @param to The target DLTensor.                                                              \
     * @param stream Optional stream object.                                                       \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*CopyDataFromTo)(const DLTensor *from, DLTensor *to, TVMStreamHandle stream);             \
                                                                                                   \
    /**                                                                                            \
     * @brief Create a new stream of execution.                                                    \
     * @param dev_id The device id to perform operation.                                           \
     * @return the allocated stream handle, NULL if fail                                           \
     */                                                                                            \
    TVMStreamHandle (*CreateStream)(int dev_id);                                                   \
                                                                                                   \
    /**                                                                                            \
     * @brief Free a stream of execution.                                                          \
     * @param dev_id The device id to perform operation.                                           \
     * @param stream The pointer to be freed.                                                      \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*FreeStream)(int dev_id, TVMStreamHandle stream);                                         \
                                                                                                   \
    /**                                                                                            \
     * @brief Synchronize the stream.                                                              \
     * @param dev_id The device id to perform operation.                                           \
     * @param stream The stream to be sync.                                                        \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*StreamSync)(int dev_id, TVMStreamHandle stream);                                         \
                                                                                                   \
    /**                                                                                            \
     * @brief Set the current stream.                                                              \
     * @param dev_id The device id to perform operation.                                           \
     * @param stream The stream to be set.                                                         \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*SetStream)(int dev_id, TVMStreamHandle stream);                                          \
                                                                                                   \
    /**                                                                                            \
     * @brief Get the current stream.                                                              \
     * @return the stream handle, NULL if fail                                                     \
     */                                                                                            \
    TVMStreamHandle (*GetStream)();                                                                \
                                                                                                   \
    /**                                                                                            \
     * @brief Synchronize 2 streams of execution.                                                  \
     * @param dev_id The device id to perform operation.                                           \
     * @param event_src The source stream to synchronize.                                          \
     * @param event_dst The destination stream to synchronize.                                     \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*SyncStreamFromTo)(int dev_id, TVMStreamHandle event_src, TVMStreamHandle event_dst);     \
                                                                                                   \
    /**                                                                                            \
     * @brief Allocate temporal workspace for backend execution.                                   \
     * @param dev_id The device id to perform operation.                                           \
     * @param nbytes The size to be allocated.                                                     \
     * @param type_hint The type of elements.                                                      \
     * @return allocated handle, NULL if fail                                                      \
     */                                                                                            \
    void *(*AllocWorkspace)(int dev_id, size_t nbytes, DLDataType type_hint);                      \
                                                                                                   \
    /**                                                                                            \
     * @brief Free temporal workspace in backend execution.                                        \
     * @param dev_id The device id to perform operation.                                           \
     * @param ptr The pointer to be freed.                                                         \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*FreeWorkspace)(int dev_id, void *ptr);                                                   \
                                                                                                   \
    /**                                                                                            \
     * @brief Free the device API instance                                                         \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*Release)(DeviceAPI * d);

/*------------------------End of the interface in device api--------------------------------------*/

/**
 * @brief The DeviceAPI is just a single instance interface.
 */
struct DeviceAPI {
    DEVICE_API_INTERFACE
};

/**
 * @brief Get the device api instance for the given device type.
 * @param deviceType device type
 * @param out_device_api The pointer to receive the point.
 * @return 0 if successful
 */
int TVM_RT_WASM_DeviceAPIGet(DLDeviceType deviceType, DeviceAPI **out_device_api);

/**
 * @brief Destroy all device api instance.
 * @note It only be used at runtime destructor.
 */
void TVM_RT_WASM_DeviceReleaseAll();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TVM_RT_WASM_CORE_DEVICE_DEVICE_API_H_INCLUDE_
