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
     * @return The allocated device pointer.  return NULL if fail.                                 \
     */                                                                                            \
    void *(*AllocDataSpace)(int dev_id, size_t nbytes);                                            \
                                                                                                   \
    /**                                                                                            \
     * @brief Free a data space on device.                                                         \
     * @param dev_id The device_id to perform operation.                                           \
     * @param ptr The data pointer.                                                                \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*FreeDataSpace)(int dev_id, void *ptr);                                                   \
                                                                                                   \
    /**                                                                                            \
     * @brief Copy data from this device to cpu.                                                   \
     * @param from The source data handle.                                                         \
     * @param to The target data handle.                                                           \
     * @param nbytes The number of bytes to copy.                                                  \
     * @param from_offset The offset for source data handle.                                       \
     * @param to_offset The offset for target data handle.                                         \
     * @param stream Optional stream object.                                                       \
     * @param from_dev_id The source data device id.                                               \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*CopyDataFromDeviceToCPU)(const void *from, void *to, size_t nbytes, size_t from_offset,  \
                                   size_t to_offset, TVMStreamHandle stream, int from_dev_id);     \
                                                                                                   \
    /**                                                                                            \
     * @brief Copy data from cpu to this device.                                                   \
     * @param from The source data handle.                                                         \
     * @param to The target data handle.                                                           \
     * @param nbytes The number of bytes to copy.                                                  \
     * @param from_offset The offset for source data handle.                                       \
     * @param to_offset The offset for target data handle.                                         \
     * @param stream Optional stream object.                                                       \
     * @param to_dev_id The target data device id.                                                 \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*CopyDataFromCPUToDevice)(const void *from, void *to, size_t nbytes, size_t from_offset,  \
                                   size_t to_offset, TVMStreamHandle stream, int to_dev_id);       \
                                                                                                   \
    /**                                                                                            \
     * @brief Copy data from one device to device (must the same device).                          \
     * @param from The source data handle.                                                         \
     * @param to The target data handle.                                                           \
     * @param nbytes The number of bytes to copy.                                                  \
     * @param from_offset The offset for source data handle.                                       \
     * @param to_offset The offset for target data handle.                                         \
     * @param stream Optional stream object.                                                       \
     * @param from_dev_id The source data device id.                                               \
     * @param to_dev_id The target data device id.                                                 \
     * @return 0 if successful                                                                     \
     */                                                                                            \
    int (*CopyDataFromDeviceToDevice)(const void *from, void *to, size_t nbytes,                   \
                                      size_t from_offset, size_t to_offset,                        \
                                      TVMStreamHandle stream, int from_dev_id, int to_dev_id);     \
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
     * @brief Allocate temporal workspace for backend execution.                                   \
     * @param dev_id The device id to perform operation.                                           \
     * @param nbytes The size to be allocated.                                                     \
     * @return allocated handle, NULL if fail                                                      \
     */                                                                                            \
    void *(*AllocWorkspace)(int dev_id, size_t nbytes);                                            \
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
