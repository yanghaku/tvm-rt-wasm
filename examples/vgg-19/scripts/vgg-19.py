import argparse
import os
import sys
import tvm
from tvm import relay, runtime
from tvm.target import Target
from tvm.relay import testing


# the function for change binary data to c source code
def data2c(in_file, out_file, var_name):
    len_name = var_name + '_len'
    with open(in_file, "rb") as in_f:
        x = bytes(in_f.read())

    out_str = 'unsigned int ' + len_name + '=' + str(len(x)) + ';\n'
    out_str += 'unsigned char ' + var_name + '[]={'
    for i in x:
        out_str += hex(i) + ","
    out_str += "};"

    with open(out_file, "w") as f:
        f.write(out_str)


def build_module(opts):
    batch_size = 1
    mod, params = testing.vgg.get_workload(
        num_layers=19, batch_size=batch_size, dtype="float32"
    )

    host = "llvm --system-lib"
    if opts.runtime == 'wasm':
        host += ' -mtriple=wasm32-unknown-wasm'
    if opts.target == "cpu":
        target = Target(host)
    else:
        target = Target("cuda -device=1050ti", host=host)
    print("build lib target = '", target, "'; runtime = '", host, "'")

    with tvm.transform.PassContext(opt_level=0):
        factory = relay.build(mod, target=target, params=params)

    factory.get_lib().export_library(os.path.join(opts.out_dir, "graph.tar"))

    json_str = "graph.json"
    params_str = "graph.params"
    json_path = os.path.join(opts.out_dir, json_str)
    param_path = os.path.join(opts.out_dir, params_str)
    with open(json_path, "w") as f_graph:
        f_graph.write(factory.get_graph_json())
    with open(param_path, "wb") as f_params:
        f_params.write(runtime.save_param_dict(factory.get_params()))

    data2c(json_path, os.path.join(opts.out_dir, json_str + ".c"), json_str.replace('.', '_'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="./lib")
    parser.add_argument("--target", default="cpu", help="target device")
    parser.add_argument("--runtime", default="wasm", help="native or wasm")
    opt = parser.parse_args()

    build_module(opt)
    print("build module success!", file=sys.stderr)
