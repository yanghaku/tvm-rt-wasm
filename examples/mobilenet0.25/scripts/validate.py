import os.path

from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
import numpy as np
from mxnet.ndarray import array, op
import sys

print("run native mobilenet_v0.25")

block = get_model("mobilenet0.25", pretrained=True)
image_filename = os.path.join(sys.path[0], "../lib/cat.png")
image = Image.open(image_filename).resize((224, 224))


def transform_image(img):
    img = np.array(img) - np.array([123.0, 117.0, 104.0])
    img /= np.array([58.395, 57.12, 57.375])
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    return img


x = transform_image(image)

print("input = ", x.shape)

output = op.softmax(block(array(x))).asnumpy()
print("output = ", output.shape)

y = output.reshape(1000)

print(y.shape)
for i in range(len(y)):
    p = int(y[i] * 1000)
    if p != 0:
        print(i, p)

print("max = out[", np.argmax(y), "] = ", np.max(y))

# 151 10
# 186 7
# 259 26
# 263 62
# 264 2
# 271 1
# 273 1
# 274 5
# 277 280
# 278 590
# 280 5
# 287 1
