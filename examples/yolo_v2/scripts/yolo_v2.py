import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import sys
import tvm
from tvm import runtime, relay
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
from tvm.target import Target
from tvm.contrib import graph_executor


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


MODEL_NAME = "yolov2"

CFG_NAME = MODEL_NAME + ".cfg"
WEIGHTS_NAME = MODEL_NAME + ".weights"
REPO_URL = "https://github.com/dmlc/web-data/blob/main/darknet/"
CFG_URL = REPO_URL + "cfg/" + CFG_NAME + "?raw=true"
WEIGHTS_URL = "https://pjreddie.com/media/files/" + WEIGHTS_NAME

cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

# Download and Load darknet library
if sys.platform in ["linux", "linux2"]:
    DARKNET_LIB = "libdarknet2.0.so"
    DARKNET_URL = REPO_URL + "lib/" + DARKNET_LIB + "?raw=true"
elif sys.platform == "darwin":
    DARKNET_LIB = "libdarknet_mac2.0.so"
    DARKNET_URL = REPO_URL + "lib_osx/" + DARKNET_LIB + "?raw=true"
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode("utf-8"), weights_path.encode("utf-8"), 0)
dtype = "float32"
batch_size = 1


def build_module(opts):
    data = np.empty([batch_size, net.c, net.h, net.w], dtype)
    shape = {"data": data.shape}
    print("Converting darknet to relay functions...")
    mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

    host = "llvm --system-lib"
    if not opts.test and opts.runtime == 'wasm':
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

    print("Compiling the model...")
    with tvm.transform.PassContext(opt_level=3):
        factory = relay.build(mod, target=target, params=params)

    if opts.test:
        return factory

    if not os.path.exists(opts.out_dir):
        os.makedirs(opts.out_dir)
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


def test(opts):
    lib = build_module(opts)
    dev = tvm.device(opts.target, 0)
    m = graph_executor.GraphModule(lib["default"](dev))

    test_image = "dog.jpg"
    print("Loading the test image...")
    img_url = REPO_URL + "data/" + test_image + "?raw=true"
    img_path = download_testdata(img_url, test_image, "data")
    data = tvm.relay.testing.darknet.load_image(img_path, net.w, net.h)
    p = "/home/yb/code/tvm-rt-wasm/build_cuda/examples/lib/yolo_v2/dog.bin"
    with open(p, "rb") as f:
        b = f.read()
        print(len(b))
        data = np.frombuffer(b, dtype=np.float32).reshape(data.shape)

    import time
    st = time.time()

    # set inputs
    m.set_input("data", tvm.nd.array(data.astype(dtype)))
    print("data.shape = ", data.shape)
    # execute
    print("Running the test image...")

    # detection
    # thresholds
    thresh = 0.5
    nms_thresh = 0.45

    m.run()
    # get outputs
    tvm_out = []

    print(m.get_output(1).numpy())
    print(m.get_output(2).numpy())
    layer_out = {"type": "Region"}
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).numpy()
    layer_out["biases"] = m.get_output(1).numpy()
    out_shape = (layer_attr[0], layer_attr[1] // layer_attr[0], layer_attr[2], layer_attr[3])
    print("out_shape = ", out_shape)
    layer_out["output"] = m.get_output(0).numpy().reshape(out_shape)
    layer_out["classes"] = layer_attr[4]
    layer_out["coords"] = layer_attr[5]
    layer_out["background"] = layer_attr[6]
    tvm_out.append(layer_out)

    print("time = ", (time.time() - st) * 1000.0, "ms")

    if opts.test_save:
        np.set_printoptions(threshold=100000000, precision=10, suppress=True)
        with open("test.res", "w") as f:
            for i in range(3):
                n = m.get_output(i).numpy()
                print("output", i, "shape=", n.shape, file=f)
                print(n, file=f)
        return

    # do the detection and bring up the bounding boxes
    img = tvm.relay.testing.darknet.load_image_color(img_path)
    _, im_h, im_w = img.shape
    print("img shape = ", img.shape)
    dets = tvm.relay.testing.yolo_detection.fill_network_boxes(
        (net.w, net.h), (im_w, im_h), thresh, 1, tvm_out
    )
    last_layer = net.layers[net.n - 1]
    tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

    coco_name = "coco.names"
    coco_url = REPO_URL + "data/" + coco_name + "?raw=true"
    font_name = "arial.ttf"
    font_url = REPO_URL + "data/" + font_name + "?raw=true"
    coco_path = download_testdata(coco_url, coco_name, module="data")
    font_path = download_testdata(font_url, font_name, module="data")

    with open(coco_path) as f:
        content = f.readlines()

    names = [x.strip() for x in content]

    tvm.relay.testing.yolo_detection.show_detections(img, dets, thresh, names, last_layer.classes)
    tvm.relay.testing.yolo_detection.draw_detections(
        font_path, img, dets, thresh, names, last_layer.classes
    )
    plt.imshow(img.transpose(1, 2, 0))
    plt.show()


def build_inputs(opts):
    test_image = "dog.jpg"
    img_url = REPO_URL + "data/" + test_image + "?raw=true"
    img_path = download_testdata(img_url, test_image, "data")
    data = tvm.relay.testing.darknet.load_image(img_path, net.w, net.h)
    with open(opts.out_dir + "/dog.bin", "wb") as f:
        f.write(data.astype(np.float32).tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default="./lib")
    parser.add_argument("--target", default="cpu", help="target device")
    parser.add_argument("--runtime", default="wasm", help="native or wasm")
    parser.add_argument("--test", default=False, type=bool, help="if test or build")
    parser.add_argument("--test-save", default=False, type=bool, help="save test results")
    opt = parser.parse_args()

    if opt.test:
        test(opt)
    else:
        build_module(opt)
        build_inputs(opt)
        print("build module success!", file=sys.stderr)
