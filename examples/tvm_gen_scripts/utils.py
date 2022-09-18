import argparse
from tvm.target import Target
from model_info import model_choices


# get target from the given options
def get_tvm_target(opts):
    host = "llvm --system-lib"
    if opts.runtime == 'wasm':
        host += ' -mtriple=wasm32-wasi -mattr=+simd128,+bulk-memory'
    if opts.target == "cpu":
        t = Target(host)
    else:
        t = Target(opts.target, host=host)
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
    parser.add_argument("--target", default="cpu", help="target device")
    parser.add_argument("--runtime", default="native", help="native or wasm")
    parser.add_argument("--executor", default="graph", help="executor type", choices=("graph", "aot"))
    parser.add_argument("--emit-llvm", default=True, type=bool, help="generate the llvm-ir")

    parser.add_argument("--tune", default=False, type=bool, help="tune module before build")
    parser.add_argument("--tune-log-file", default="tune.log", help="tune log file")
    parser.add_argument("--tune-n-trial", default=200, type=int, help="tune trial number")

    return parser
