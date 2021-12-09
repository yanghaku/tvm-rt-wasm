import os
import numpy as np
import tvm
import onnx
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata, download
from tvm.target import Target
from PIL import Image

model_url = "".join(
    [
        "https://github.com/onnx/models/raw/",
        "master/vision/classification/resnet/model/",
        "resnet50-v2-7.onnx",
    ]
)


def transform_image(img):
    img = np.array(img) - np.array([123.0, 117.0, 104.0])
    img /= np.array([58.395, 57.12, 57.375])
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
image_fn = os.path.join("cat.png")
download(image_url, image_fn)
image = Image.open(image_fn).resize((224, 224))
img_data = transform_image(image).astype("float32")

input_name = "data"
shape_dict = {input_name: img_data.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

host = "llvm --system-lib"
target = Target("cuda -device=1050ti", host=host)
print("build lib target = '", target, "'; runtime = '", host, "'")

with tvm.transform.PassContext(opt_level=3):
    factory = relay.build(mod, target=target, params=params)

# create runtime
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(factory["default"](dev))
data_tvm = tvm.nd.array(img_data)
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
