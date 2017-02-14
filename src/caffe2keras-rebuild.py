from __future__ import division, print_function
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import os
import re


class LocalResponseNormalization(Layer):
    
    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_dim_ordering == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        half_n = self.n // 2
        squared = K.square(x)
        pooled = K.pool2d(squared, (half_n, half_n), strides=(1, 1),
                         border_mode="same", pool_mode="avg")
        if K.image_dim_ordering == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom
    
    def get_output_shape_for(self, input_shape):
        return input_shape

def transform_conv_weight(W):
    # for non FC layers, do this because Keras/Theano does convolution vs 
    # Caffe correlation
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W

def transform_fc_weight(W):
    return W.T

def preprocess_image(img, resize_wh, mean_image):
    # resize
    img4d = imresize(img, (resize_wh, resize_wh))
    img4d = img4d.astype("float32")
    # BGR -> RGB
    img4d = img4d[:, :, ::-1]
    # swap axes to theano mode
    img4d = np.transpose(img4d, (2, 0, 1))
    # add batch dimension
    img4d = np.expand_dims(img4d, axis=0)
    # subtract mean image
    img4d -= mean_image
    # clip to uint
    img4d = np.clip(img4d, 0, 255).astype("uint8")
    return img4d
    
DATA_DIR = "../data/vgg-cnn"
CAT_IMAGE = os.path.join(DATA_DIR, "cat.jpg")
MEAN_IMAGE = os.path.join(DATA_DIR, "mean_image.npy")
CAFFE_WEIGHTS_DIR = os.path.join(DATA_DIR, "saved-weights")
LABEL_FILE = os.path.join(DATA_DIR, "caffe2keras-labels.txt")
KERAS_MODEL_FILE = os.path.join(DATA_DIR, "vggcnn-keras.h5")
RESIZE_WH = 224

# caffe model layers (reference)
CAFFE_LAYER_NAMES = [
    "data", 
    "conv1", "norm1", "pool1",
    "conv2", "pool2",
    "conv3",
    "conv4",
    "conv5", "pool5",
    "fc6",
    "fc7",
    "prob"
]
CAFFE_LAYER_SHAPES = {
    "data" : (10, 3, 224, 224),
    "conv1": (10, 96, 109, 109),
    "norm1": (10, 96, 109, 109),
    "pool1": (10, 96, 37, 37),
    "conv2": (10, 256, 33, 33),
    "pool2": (10, 256, 17, 17),
    "conv3": (10, 512, 17, 17),
    "conv4": (10, 512, 17, 17),
    "conv5": (10, 512, 17, 17),
    "pool5": (10, 512, 6, 6),
    "fc6"  : (10, 4096),
    "fc7"  : (10, 4096),
    "fc8"  : (10, 1000),
    "prob" : (10, 1000)
}

print("caffe:")
for layer_name in CAFFE_LAYER_NAMES:
    print(layer_name, CAFFE_LAYER_SHAPES[layer_name])

# data (10, 3, 224, 224)
# conv1 (10, 96, 109, 109)
# norm1 (10, 96, 109, 109)
# pool1 (10, 96, 37, 37)
# conv2 (10, 256, 33, 33)
# pool2 (10, 256, 17, 17)
# conv3 (10, 512, 17, 17)
# conv4 (10, 512, 17, 17)
# conv5 (10, 512, 17, 17)
# pool5 (10, 512, 6, 6)
# fc6 (10, 4096)
# fc7 (10, 4096)
# prob (10, 1000)

# set theano dimension ordering
# NOTE: results match Caffe using Theano backend only
K.set_image_dim_ordering("th")

# load weights
W_conv1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1.npy")))
b_conv1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1.npy"))

W_conv2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2.npy")))
b_conv2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2.npy"))

W_conv3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3.npy")))
b_conv3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3.npy"))

W_conv4 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4.npy")))
b_conv4 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4.npy"))

W_conv5 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv5.npy")))
b_conv5 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv5.npy"))

W_fc6 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc6.npy")))
b_fc6 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc6.npy"))

W_fc7 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc7.npy")))
b_fc7 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc7.npy"))

W_fc8 = transform_fc_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_fc8.npy")))
b_fc8 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_fc8.npy"))

# define network
data = Input(shape=(3, 224, 224), name="DATA")

conv1 = Convolution2D(96, 7, 7, subsample=(2, 2),
                     weights=(W_conv1, b_conv1))(data)
conv1 = Activation("relu", name="CONV1")(conv1)

norm1 = LocalResponseNormalization(name="NORM1")(conv1)

pool1 = ZeroPadding2D(padding=(0, 2, 0, 2))(norm1)
pool1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="POOL1")(pool1)

conv2 = Convolution2D(256, 5, 5, weights=(W_conv2, b_conv2))(pool1)
conv2 = Activation("relu", name="CONV2")(conv2)

pool2 = ZeroPadding2D(padding=(0, 1, 0, 1))(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="POOL2")(pool2)

conv3 = ZeroPadding2D(padding=(0, 2, 0, 2))(pool2)
conv3 = Convolution2D(512, 3, 3, weights=(W_conv3, b_conv3))(conv3)
conv3 = Activation("relu", name="CONV3")(conv3)

conv4 = ZeroPadding2D(padding=(0, 2, 0, 2))(conv3)
conv4 = Convolution2D(512, 3, 3, weights=(W_conv4, b_conv4))(conv4)
conv4 = Activation("relu", name="CONV4")(conv4)

conv5 = ZeroPadding2D(padding=(0, 2, 0, 2))(conv4)
conv5 = Convolution2D(512, 3, 3, weights=(W_conv5, b_conv5))(conv5)
conv5 = Activation("relu", name="CONV5")(conv5)

pool5 = ZeroPadding2D(padding=(0, 1, 0, 1))(conv5)
pool5 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name="POOL5")(pool5)

fc6 = Flatten()(pool5)
fc6 = Dense(4096, weights=(W_fc6, b_fc6))(fc6)
fc6 = Activation("relu", name="FC6")(fc6)

fc7 = Dense(4096, weights=(W_fc7, b_fc7))(fc6)
fc7 = Activation("relu", name="FC7")(fc7)

fc8 = Dense(1000, weights=(W_fc8, b_fc8), name="FC8")(fc7)
prob = Activation("softmax", name="PROB")(fc8)

model = Model(input=[data], output=[prob])

model.compile(optimizer="adam", loss="categorical_crossentropy")

print("keras")
for layer in model.layers:
    print(layer.name, layer.output_shape)

# DATA (None, 3, 224, 224)
# convolution2d_1 (None, 96, 109, 109)
# CONV1 (None, 96, 109, 109)
# NORM1 (None, 96, 109, 109)
# zeropadding2d_1 (None, 96, 111, 111)
# POOL1 (None, 96, 37, 37)
# convolution2d_2 (None, 256, 33, 33)
# CONV2 (None, 256, 33, 33)
# zeropadding2d_2 (None, 256, 34, 34)
# POOL2 (None, 256, 17, 17)
# zeropadding2d_3 (None, 256, 19, 19)
# convolution2d_3 (None, 512, 17, 17)
# CONV3 (None, 512, 17, 17)
# zeropadding2d_4 (None, 512, 19, 19)
# convolution2d_4 (None, 512, 17, 17)
# CONV4 (None, 512, 17, 17)
# zeropadding2d_5 (None, 512, 19, 19)
# convolution2d_5 (None, 512, 17, 17)
# CONV5 (None, 512, 17, 17)
# zeropadding2d_6 (None, 512, 18, 18)
# POOL5 (None, 512, 6, 6)
# flatten_1 (None, 18432)
# dense_1 (None, 4096)
# FC6 (None, 4096)
# dense_2 (None, 4096)
# FC7 (None, 4096)
# FC8 (None, 1000)
# PROB (None, 1000)

# prediction
id2label = {}
flabel = open(LABEL_FILE, "rb")
for line in flabel:
    lid, lname = line.strip().split("\t")
    id2label[int(lid)] = lname
flabel.close()

mean_image = np.load(MEAN_IMAGE)
image = plt.imread(CAT_IMAGE)
img4d = preprocess_image(image, RESIZE_WH, mean_image)

print(image.shape, mean_image.shape, img4d.shape)

preds = model.predict(img4d)[0]
print(np.argmax(preds))
# 281

top_preds = np.argsort(preds)[::-1][0:10]
# array([281, 285, 282, 277, 287, 284, 283, 263, 387, 892])

pred_probas = [(x, id2label[x], preds[x]) for x in top_preds]
print(pred_probas)

print("Saving model...")
model.save(KERAS_MODEL_FILE)

