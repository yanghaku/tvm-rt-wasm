import { DLDevice } from "./dlpack";

export class GraphExecutor {
    private wasm_instance: WebAssembly.Instance;
    private handle: number;

    constructor(wasm_instance: WebAssembly.Instance, graph_json: string, module_handle: WebAssembly.Module, device: DLDevice) {
        if (device == null) {
            device = DLDevice.CPU(); // default is CPU
        }
        this.wasm_instance = wasm_instance;

        wasm_instance.exports["malloc"]
    }
};
