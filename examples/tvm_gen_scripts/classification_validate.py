import os.path

import tvm.runtime.ndarray
from tvm.contrib.download import download_testdata
import numpy as np
from PIL import Image

from module_process import build_ir_module, run_module
from model_info import get_ir_module_from_frontend, get_model_info
from utils import get_arg_parser, get_tvm_target


def get_classification_label():
    url = "https://raw.github.com/onnx/models/main/vision/classification/synset.txt"
    path = download_testdata(url, "imagenet1000_clsid_to_human.txt", module="data")
    with open(path, "r") as f:
        return [i.strip() for i in f]


def get_classification_inputs(opts):
    # This pre_process function is copied from https://github.com/onnx/models/tree/main/vision/classification/resnet
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
    d = preprocess(img)[np.newaxis, :]
    if opts.validate:
        return d
    with open(os.path.join(opts.out_dir, "cat.bin"), "wb") as f:
        f.write(d.tobytes())


def validate_model(opts):
    ir_module, params = get_ir_module_from_frontend(opts)
    target = get_tvm_target(opts)
    m = build_ir_module(opts, ir_module, params, target)

    module_info = get_model_info(opts.model)
    inputs = tvm.runtime.ndarray.array(get_classification_inputs(opts))
    input_dict = {list(module_info.input_info.keys())[0]: inputs}

    outputs = run_module(opts, m, input_dict)

    output = outputs[0].reshape(1000)
    arg_max = np.argmax(output)

    print("max = out[", arg_max, "] = ", np.max(output))
    print("classification = ", get_classification_label()[arg_max])


if __name__ == '__main__':
    parser = get_arg_parser()
    parser.add_argument("--validate", type=bool, default=False, help="validate the model")
    opt = parser.parse_args()

    if opt.validate:
        # validate model
        opt.host_target = 'native'
        validate_model(opt)
    else:
        # just generate the inputs for classification
        get_classification_inputs(opt)
