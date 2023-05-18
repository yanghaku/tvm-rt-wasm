export const enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
}

const DLDeviceTypeStrings = ["NULL", "kDLCPU", "kDLCUDA", "kDLCUDAHost", "kDLOpenCL", "kDLVulkan", "kDLMetal", "kDLVPI", "kDLROCM", "kDLROCMHost", "kDLExtDev", "kDLCUDAManaged", "kDLOneAPI", "kDLWebGPU", "kDLHexagon"];

export class DLDevice {
    private device_type: number;
    private device_id: number;

    constructor(device_type: DLDeviceType,
        device_id: number) {
        this.device_type = device_type;
        this.device_id = device_id;
    }

    static CPU(): DLDevice {
        return new DLDevice(DLDeviceType.kDLCPU, 0);
    }

    static WebGPU(): DLDevice {
        return new DLDevice(DLDeviceType.kDLWebGPU, 0);
    }

    static SizeOf(): number {
        return 8;
    }

    write_to_memory(arr: Uint8Array): void {
        const arr32 = Uint32Array.from(arr);
        arr32[0] = this.device_type;
        arr32[1] = this.device_id;
    }

    toString(): string {
        return "DLDevice { type = " + DLDeviceTypeStrings[this.device_type] + ", id = " + this.device_id.toString() + "}";
    }
}

export const enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
}

export class DLDataType {
    private code: number;
    private bits: number;
    private lanes: number;

    constructor(code: DLDataTypeCode, bits: number, lanes: number) {
        this.code = code;
        this.bits = bits;
        this.lanes = lanes;
    }
}

// export class DLTensor {
//     data: ArrayBuffer;
//     device: DLDevice;
//     ndim: number;
//     dtype: number;
//     shape: Array<number>;
//     strides: Array<number>;
//     byte_offset: number;
// }
