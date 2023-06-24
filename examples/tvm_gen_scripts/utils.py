import argparse
from tvm.target import Target
from model_info import model_choices


# get target from the given options
def get_tvm_target(opts):
    host = "llvm"
    if opts.host_target == 'wasm32-wasi':
        host += ' -mtriple=wasm32-wasi -mattr=+simd128,+bulk-memory'
    elif opts.host_target == 'wasm32-emscripten':
        host += ' -mtriple=wasm32-emscripten -mattr=+bulk-memory'

    if opts.device_target == "cpu":
        t = Target(host, host=host)
    else:
        t = Target(opts.device_target, host=host)
    print("build lib target = '", t, "'; runtime = '", host, "'")
    return t


# the function for change binary data to c source code
def data2c(data, out_file, var_name):
    len_name = var_name + '_len'
    out_str = 'unsigned int ' + len_name + '=' + str(len(data)) + ';\n'
    out_str += 'unsigned char ' + var_name + '[]={'
    for i in bytes(data, encoding="utf-8"):
        out_str += hex(i) + ","
    out_str += "};"

    with open(out_file, "w") as f:
        f.write(out_str)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenet", help="model name to build", choices=model_choices.keys())
    parser.add_argument("-o", "--out-dir", default="./lib")
    parser.add_argument("--device-target", default="cpu", help="device target")
    parser.add_argument("--host-target", default="native",
                        help="host runtime target, e.g. native,wasm32-wasi,wasm32-emscripten")
    parser.add_argument("--executor", default="graph", help="executor type", choices=("graph", "aot"))
    parser.add_argument("--emit-llvm", default=False, type=bool, help="generate the llvm-ir")
    parser.add_argument("--dso", default=False, type=bool, help="create dynamic library")
    parser.add_argument("--dso-only", default=False, type=bool, help="create dynamic library only, (no params,json)")

    parser.add_argument("--tune", default=False, type=bool, help="tune module before build")
    parser.add_argument("--tune-log-file", default="tune.log", help="tune log file")
    parser.add_argument("--tune-n-trial", default=200, type=int, help="tune trial number")

    return parser
