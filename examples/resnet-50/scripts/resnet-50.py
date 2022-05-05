import argparse
import os
import sys
import numpy as np
import tvm
import onnx
from tvm import relay, runtime
from tvm.contrib import download
from tvm.contrib.download import download_testdata
from tvm.target import Target
from PIL import Image

model_url = "".join(
    [
        "https://github.com/onnx/models/raw/",
        "main/vision/classification/resnet/model/",
        "resnet50-v2-7.onnx",
    ]
)


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


def transform_image(img):
    img = np.array(img) - np.array([123.0, 117.0, 104.0])
    img /= np.array([58.395, 57.12, 57.375])
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


def build_module(opts):
    model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
    onnx_model = onnx.load(model_path)

    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join(opts.out_dir, "cat.png")
    download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))
    img_data = transform_image(image).astype("float32")

    input_name = "data"
    shape_dict = {input_name: img_data.shape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    host = "llvm --system-lib"
    if opts.runtime == 'wasm':
        host += ' -mtriple=wasm32-unknown-wasm'  # -mattr=+simd128
    if opts.target == "cpu":
        target = Target(host)
    else:
        # use environment variable to custom cuda device such as jetson-nano
        # e.g. "nvidia/jetson-nano"
        if 'device' in os.environ:
            device = os.environ['device']
            print("custom the cuda device:", device)
        else:
            device = 'cuda'
            print("use default host cuda device")
        target = Target(device, host=host)
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
