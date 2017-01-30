# -*- coding: utf-8 -*-
from __future__ import division, print_function
import caffe
import numpy as np
import os

CAFFE_HOME="/home/ubuntu/mnt/caffe"

MODEL_DIR = os.path.join(CAFFE_HOME, "models", "vgg_cnn_s")
MODEL_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "VGG_CNN_S.caffemodel")
MEAN_IMAGE = os.path.join(MODEL_DIR, "VGG_mean.binaryproto")

OUTPUT_DIR = "../../data/vgg-cnn-weights"

caffe.set_mode_cpu()
net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)

for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(OUTPUT_DIR, "W_{:s}.npy".format(k)), v[0].data)
    np.save(os.path.join(OUTPUT_DIR, "b_{:s}.npy".format(k)), v[1].data)

# layer    W.shape    b.shape
#conv1 (96, 3, 7, 7) (96,)
#conv2 (256, 96, 5, 5) (256,)
#conv3 (512, 256, 3, 3) (512,)
#conv4 (512, 512, 3, 3) (512,)
#conv5 (512, 512, 3, 3) (512,)
#fc6 (4096, 18432) (4096,)
#fc7 (4096, 4096) (4096,)
#fc8 (1000, 4096) (1000,)

blob = caffe.proto.caffe_pb2.BlobProto()
with open(MEAN_IMAGE, 'rb') as fmean:
    mean_data = fmean.read()
blob.ParseFromString(mean_data)
mu = np.array(caffe.io.blobproto_to_array(blob))
print("Mean image:", mu.shape)
np.save(os.path.join(OUTPUT_DIR, "mean_image.npy"), mu)

#Mean image: (1, 3, 224, 224)

for layer_name, blob in net.blobs.iteritems():
    print(layer_name, blob.data.shape)

#data (10, 3, 224, 224)
#conv1 (10, 96, 109, 109)
#norm1 (10, 96, 109, 109)
#pool1 (10, 96, 37, 37)
#conv2 (10, 256, 33, 33)
#pool2 (10, 256, 17, 17)
#conv3 (10, 512, 17, 17)
#conv4 (10, 512, 17, 17)
#conv5 (10, 512, 17, 17)
#pool5 (10, 512, 6, 6)
#fc6 (10, 4096)
#fc7 (10, 4096)
#fc8 (10, 1000)
#prob (10, 1000)