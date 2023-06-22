import tvm
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
import numpy as np
from PIL import Image
import argparse


def get_classification_inputs():
    def preprocess(img_data):  # img_data data layout is CHW
        mean_vec = np.array([0.485, 0.456, 0.406]).astype(np.float64)
        stddev_vec = np.array([0.229, 0.224, 0.225]).astype(np.float64)
        norm_img_data = np.zeros(img_data.shape).astype(np.float64)
        for i in range(img_data.shape[0]):
            # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data.astype(np.float32)

    img_path = download_testdata("https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg", "cat.jpg", "data")
    img = np.asarray(Image.open(img_path).resize((224, 224))).transpose((2, 0, 1))
    return preprocess(img)[np.newaxis, :]


def run(opts):
    lib = tvm.runtime.load_module(opts.lib)
    json = open(opts.json, "r").read()
    param_bytes = open(opts.param, "rb").read()
    dev = tvm.device('cpu', 0)
    model = graph_executor.create(json, lib, dev)
    model.load_params(param_bytes)
    model.set_input(opts.input_name, get_classification_inputs())
    model.run()
    return model.get_output(0, tvm.nd.empty((1, 1000), "float32", dev)).asnumpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="shared library path")
    parser.add_argument("json", help="graph json path")
    parser.add_argument("param", help="graph params path")
    parser.add_argument("--input-name", default='input', help="graph params path")
    o = parser.parse_args()

    output = run(o)
    d = np.argmax(output[0])
    print(d, output[0][d])
