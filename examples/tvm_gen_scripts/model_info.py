import onnx
from tvm import relay
from tvm.contrib.download import download_testdata


class ModelInfo:
    def __init__(self, key, full_name, input_info, url, from_frontend="onnx"):
        self.key = key
        self.full_name = full_name
        self.input_info = input_info
        self.url = "https://github.com/onnx/models/raw/main/" + url + "/model/" + full_name
        self.from_frontend = from_frontend


model_choices = {
    "mobilenet": ModelInfo("mobilenet", "mobilenetv2-12.onnx", {"input": (1, 3, 224, 224)},
                           "vision/classification/mobilenet"),
    "vgg-16": ModelInfo("vgg-16", "vgg16-7.onnx", {"data": (1, 3, 224, 224)},
                        "vision/classification/vgg"),
    "resnet-50": ModelInfo("resnet-50", "resnet50-v2-7.onnx", {"data": (1, 3, 224, 224)},
                           "vision/classification/resnet"),
    "yolo-v4": ModelInfo("yolo-v4", "yolov4.onnx", {"0": (1, 416, 416, 3)},
                         "vision/object_detection_segmentation/yolov4")
}


def get_model_info(name):
    if name in model_choices:
        return model_choices[name]
    raise "unsupported module name"


def get_module_frontend(opts):
    model_info = get_model_info(opts.model)

    model_local_path = download_testdata(model_info.url, model_info.full_name, module=model_info.from_frontend)

    if model_info.from_frontend == "onnx":
        onnx_module = onnx.load(model_local_path)
        mod, params = relay.frontend.from_onnx(onnx_module, shape=model_info.input_info)
    else:
        # todo: add new frontend such as tf,pytorch,mxnet
        raise "unsupported frontend"

    return mod, params
