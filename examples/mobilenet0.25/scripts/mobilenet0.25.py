# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
from tvm import relay
import tvm
from tvm import runtime
from tvm.target import Target
import sys


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
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenet0.25", pretrained=True)
    shape_dict = {"data": dshape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)

    func = mod["main"]
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )

    host = "llvm --system-lib"
    if opts.runtime == 'wasm':
        host += ' -mtriple=wasm32-unknown-wasm'
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

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        factory = relay.build(
            func, target=target, params=params
        )

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
    data2c(param_path, os.path.join(opts.out_dir, params_str + ".c"), params_str.replace('.', '_'))


def build_inputs(opts):
    from tvm.contrib import download
    from PIL import Image
    import numpy as np

    # Download test image
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join(opts.out_dir, "cat.jpg")
    download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(img):
        img = np.array(img) - np.array([123.0, 117.0, 104.0])
        img /= np.array([58.395, 57.12, 57.375])
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]
        return img

    x = transform_image(image)
    print("cat.bin.shape = ", x.shape)
    with open(os.path.join(opts.out_dir, "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="./lib")
    parser.add_argument("--target", default="cpu", help="target device")
    parser.add_argument("--runtime", default="wasm", help="native or wasm")
    opt = parser.parse_args()

    build_module(opt)
    print("build module success", file=sys.stderr)
    build_inputs(opt)
    print("build inputs success", file=sys.stderr)
