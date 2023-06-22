type GraphHandle = number;

type Pointer = number;

declare interface TvmRtWasmModule extends EmscriptenModule {
    _TVM_RT_WASM_GraphExecutorCreate(
        graph_json: Pointer,
        module_handle: number,
        devices: Pointer,
        num_devices: number,
    ): GraphHandle;

    _TVM_RT_WASM_GraphExecutorLoadParamsFromFile(graph_handle: GraphHandle, file_name: Pointer): number;

    _TVM_RT_WASM_GraphExecutorGetNumOfNodes(graph_handle: GraphHandle): number;

    _TVM_RT_WASM_GraphExecutorGetNumInputs(graph_handle: GraphHandle): number;

    _TVM_RT_WASM_GraphExecutorGetNumOutputs(graph_handle: GraphHandle): number;

    _TVM_RT_WASM_JS_GraphExecutorGetNodeName(graph_handle: GraphHandle, node_id: number): Pointer;

    _TVM_RT_WASM_GraphExecutorGetInputIndex(graph_handle: GraphHandle, name: Pointer): number;

    _TVM_RT_WASM_GraphExecutorGetOutputIndex(graph_handle: GraphHandle, name: Pointer): number;

    _TVM_RT_WASM_GraphExecutorSetInput(graph_handle: GraphHandle, index: number, data: Pointer): number;

    _TVM_RT_WASM_GraphExecutorGetOutput(graph_handle: GraphHandle, index: number, data: Pointer): number;

    _TVM_RT_WASM_GraphExecutorRun(graph_handle: GraphHandle): number;

    _TVM_RT_WASM_GraphExecutorDestory(graph_handle: GraphHandle): number;

    _TVMGetLastError(): Pointer;

    _TVMSynchronize(device_type: number, device_id: number, stream: number): number;

    FS: {
        getPath(node: FS.FSNode): string;

        createDataFile(
            parent: string | FS.FSNode,
            name: string,
            data: ArrayBufferView,
            canRead: boolean,
            canWrite: boolean,
            canOwn: boolean,
        ): FS.FSNode;
    },

    stackAlloc(size: number): Pointer;
    stackSave(): Pointer;
    stackRestore(ptr: Pointer): void;

    UTF8ToString(ptr: number, maxBytesToRead?: number): string;
    stringToUTF8(str: string, outPtr: number, maxBytesToRead?: number): void;
    stringToNewUTF8(str: string): number;
    stringToUTF8OnStack(str: string): number;
}

declare function tvm_rt_module_create(mod?: any): Promise<TvmRtWasmModule>;

export { tvm_rt_module_create, TvmRtWasmModule, GraphHandle, Pointer };
