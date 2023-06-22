import { DLDevice } from "./dlpack";
import { tvm_rt_module_create, TvmRtWasmModule, GraphHandle, Pointer } from '../build/tvm-rt-module';

export class GraphExecutor {
    private wasm: TvmRtWasmModule;
    private graph: GraphHandle;
    private device: DLDevice;
    private nodes_names: Map<String, number>;

    private constructor(wasm: TvmRtWasmModule, device: DLDevice, graph: GraphHandle) {
        this.wasm = wasm;
        this.graph = graph;
        this.device = device;
        this.nodes_names = new Map();
        console.log(wasm);
    }

    public static async create(
        module_handle: WebAssembly.Module,
        device: DLDevice | string | null,
        graph_json: string,
        graph_params: ArrayBufferView,
    ): Promise<GraphExecutor> {
        // todo: module handle
        if (module_handle != null) {
            console.error("cannot support module now");
        }

        if (device == null) {
            device = DLDevice.CPU(); // default is CPU
        } else if (typeof device == 'string') {
            const d = device.toLowerCase();
            if (d.includes("gpu")) {
                device = DLDevice.WebGPU();
            } else if (d.includes("cpu")) {
                device = DLDevice.CPU();
            } else {
                return Promise.reject("Unsupported device type: " + device);
            }
        }

        const wasm = await tvm_rt_module_create();
        const json_ptr = wasm.stringToNewUTF8(graph_json);
        const dev_ptr = wasm.stackAlloc(DLDevice.SizeOf());
        device.write_to_memory(wasm.HEAPU8.subarray(dev_ptr));
        const graph = wasm._TVM_RT_WASM_GraphExecutorCreate(json_ptr, 0, dev_ptr, 1);
        if (graph == null) {
            return Promise.reject(wasm.UTF8ToString(wasm._TVMGetLastError()));
        }

        // create a mem file and load params
        const f_path = "param_" + Math.random().toString();
        const f_node = wasm.FS.createDataFile("/", f_path, graph_params, true, false, false);
        const full_path = wasm.FS.getPath(f_node);
        const path_ptr = wasm.stringToUTF8OnStack(full_path);
        const res = wasm._TVM_RT_WASM_GraphExecutorLoadParamsFromFile(graph, path_ptr);
        if (res != 0) {
            return Promise.reject(wasm.UTF8ToString(wasm._TVMGetLastError()));
        }
        return Promise.resolve(new GraphExecutor(wasm, device, graph));
    }
};
