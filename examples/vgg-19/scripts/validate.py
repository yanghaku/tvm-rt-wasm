import argparse
import os
import sys
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.target import Target
from tvm.relay import testing
from tvm.contrib.download import download
from PIL import Image
import numpy as np


def transform_image(img):
    img = np.array(img) - np.array([123.0, 117.0, 104.0])
    img /= np.array([58.395, 57.12, 57.375])
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


def get_img():
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join("cat.png")
    download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))
    img_data = transform_image(image).astype("float32")
    return img_data


def build_module_run(opts):
    batch_size = 1
    mod, params = testing.vgg.get_workload(
        num_layers=19, batch_size=batch_size, dtype="float32"
    )

    host = "llvm --system-lib"
    if opts.target == "cpu":
        target = Target(host)
    else:
        target = Target("cuda -device=1050ti", host=host)
    print("build lib target = '", target, "'; runtime = '", host, "'")

    with tvm.transform.PassContext(opt_level=0):
        factory = relay.build(mod, target=target, params=params)

    # create runtime
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(factory["default"](dev))
    data_tvm = tvm.nd.array(get_img())
    module.set_input("data", data_tvm)

    # evaluate
    module.run()
    output = module.get_output(0).numpy()
    print("output = ", output.shape)

    y = output.reshape(1000)

    print(y.shape)
    for i in range(len(y)):
        p = int(y[i] * 10000)
        if p != 0:
            print(i, p)

    print("max = out[", np.argmax(y), "] = ", np.max(y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="cpu", help="target device")
    opt = parser.parse_args()

    build_module_run(opt)
