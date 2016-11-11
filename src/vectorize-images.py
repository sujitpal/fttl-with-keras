# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
import os

def get_next_image_loc(imgdir):
    for root, dirs, files in os.walk(imgdir):
        for name in files:
            path = os.path.join(root, name).split(os.path.sep)[::-1]
            yield (path[1], path[0])
        

def vectorize_batch(image_locs, image_dir, image_width, model,
                    fvec_x, fvec_y, fvec_f):
    Xs, ys, fs = [], [], []
    for subdir, filename in image_locs:
        # preprocess image for loading into CNN
        img = image.load_img(os.path.join(image_dir, subdir, filename), 
                             target_size=(image_width, image_width))
        img4d = image.img_to_array(img)
        img4d = np.expand_dims(img4d, axis=0)
        img4d = preprocess_input(img4d)
        Xs.append(img4d[0])
        ys.append(int(subdir))
        fs.append(filename)
    X = np.array(Xs)
    vecs = model.predict(X)
    # output shape is (10, 7, 7, 512)
    for i in range(len(Xs)):
        vec = vecs[i].flatten()
        vec_str = ",".join(["{:.5f}".format(x) for x in vec.tolist()])
        fvec_x.write("{:s}\n".format(vec_str))
        fvec_y.write("{:d}\n".format(ys[i]))
        fvec_f.write("{:s}\n".format(fs[i]))
    return len(Xs)
                
                
############################ main ############################

DATA_DIR = "../data/files"
IMAGE_DIR = os.path.join(DATA_DIR, "sample")
BATCH_SIZE = 10
IMAGE_WIDTH = 224
VEC_FILE_X = os.path.join(DATA_DIR, "images-X.txt")
VEC_FILE_Y = os.path.join(DATA_DIR, "images-y.txt")
VEC_FILE_F = os.path.join(DATA_DIR, "images-f.txt")

# load VGG-16 model
vgg16_model = VGG16(weights="imagenet", include_top=True)
model = Model(input=vgg16_model.input, 
              output=vgg16_model.get_layer("block5_pool").output)

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy")

fvec_x = open(VEC_FILE_X, "wb")
fvec_y = open(VEC_FILE_Y, "wb")
fvec_f = open(VEC_FILE_F, "wb")
batch = []
nbr_written = 0
for image_loc in get_next_image_loc(IMAGE_DIR):
    batch.append(image_loc)
    if len(batch) == 10:
        nbr_written += vectorize_batch(batch, IMAGE_DIR, IMAGE_WIDTH, model,
                                       fvec_x, fvec_y, fvec_f)
        print("Vectors generated for {:d} images...".format(nbr_written))
        batch = []
if len(batch) > 0:
    nbr_written += vectorize_batch(batch, IMAGE_DIR, IMAGE_WIDTH, model,
                                   fvec_x, fvec_y, fvec_f)
    print("Vectors generated for {:d} images, COMPLETE".format(nbr_written))

fvec_x.close()    
fvec_y.close()
fvec_f.close()