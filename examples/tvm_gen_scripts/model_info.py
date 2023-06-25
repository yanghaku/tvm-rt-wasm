class ModelInfo:
    def __init__(self, key, full_name, input_info, url, from_frontend="onnx"):
        self.key = key
        self.full_name = full_name
        self.input_info = input_info
        self.url = url
        self.from_frontend = from_frontend


model_choices = {
    "mobilenet": ModelInfo("mobilenet", "mobilenetv2-12.onnx", {"input": (1, 3, 224, 224)},
                           "vision/classification/mobilenet"),
    "vgg-16": ModelInfo("vgg-16", "vgg16-7.onnx", {"data": (1, 3, 224, 224)},
                        "vision/classification/vgg"),
    "vgg-16-int8": ModelInfo("vgg-16-int8", "vgg16-12-int8.onnx", {"data": (1, 3, 224, 224)},
                             "vision/classification/vgg"),
    "vgg-19": ModelInfo("vgg-19", "vgg19-7.onnx", {"data": (1, 3, 224, 224)},
                        "vision/classification/vgg"),
    "resnet-50": ModelInfo("resnet-50", "resnet50-v2-7.onnx", {"data": (1, 3, 224, 224)},
                           "vision/classification/resnet"),
    "resnet-50-int8": ModelInfo("resnet-50-int8", "resnet50-v1-12-int8.onnx", {"data": (1, 3, 224, 224)},
                                "vision/classification/resnet/"),
    "resnet-18": ModelInfo("resnet-18", "resnet18-v2-7.onnx", {"data": (1, 3, 224, 224)},
                           "vision/classification/resnet"),
    "resnet-34": ModelInfo("resnet-34", "resnet34-v2-7.onnx", {"data": (1, 3, 224, 224)},
                           "vision/classification/resnet"),
    "resnet-101": ModelInfo("resnet-101", "resnet101-v2-7.onnx", {"data": (1, 3, 224, 224)},
                            "vision/classification/resnet"),
    "resnet-152": ModelInfo("resnet-152", "resnet152-v2-7.onnx", {"data": (1, 3, 224, 224)},
                            "vision/classification/resnet"),
    "yolo-v4": ModelInfo("yolo-v4", "yolov4.onnx", {"0": (1, 416, 416, 3)},
                         "vision/object_detection_segmentation/yolov4"),
    "bert-large-uncased": ModelInfo("bert-large-uncased", "bert-large-uncased", '', '', "pytorch"),
    "bert-base-uncased": ModelInfo("bert-base-uncased", "bert-base-uncased", '', '', "pytorch"),
}


def get_model_info(name):
    if name in model_choices:
        return model_choices[name]
    raise "unsupported module name"


def _load_ir_module_from_onnx(opts, model_info):
    import onnx
    from tvm.contrib.download import download_testdata

    url = "https://github.com/onnx/models/raw/main/" + model_info.url + "/model/" + model_info.full_name
    model_local_path = download_testdata(url, model_info.full_name, module=model_info.from_frontend)
    onnx_module = onnx.load(model_local_path)

    if opts.executor == "relax_vm":
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        ir_module = from_onnx(onnx_module, shape_dict=model_info.input_info)
        ir_module = relax.transform.DecomposeOpsForInference()(ir_module)
        ir_module = relax.transform.LegalizeOps()(ir_module)
        return relax.frontend.detach_params(ir_module)
    elif opts.executor == "relay_vm" or opts.executor == "graph" or opts.executor == "aot":
        from tvm.relay.frontend import from_onnx
        return from_onnx(onnx_module, shape=model_info.input_info)
    else:
        raise Exception('unsupported backend type: ' + opts.executor)


def get_ir_module_from_frontend(opts):
    model_info = get_model_info(opts.model)

    if model_info.from_frontend == "onnx":
        return _load_ir_module_from_onnx(opts, model_info)
    elif model_info.from_frontend == "pytorch":
        if model_info.full_name[:4] == 'bert':
            from bert import get_bert_ir_module
            return get_bert_ir_module(model_info.full_name)
        else:
            raise Exception("unsupported model for pytorch")
    else:
        # todo: add new frontend such as tf,pytorch,mxnet
        raise Exception("unsupported frontend")
