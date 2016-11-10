# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import os

def get_next_image_loc(imgdir):
    for root, dirs, files in os.walk(imgdir):
        for name in files:
            path = os.path.join(root, name).split(os.path.sep)[::-1]
            yield (path[1], path[0])


def write_vectors(model, X, y, tag, data_dir, batch_size):
    fXout = open(os.path.join(data_dir, 
        "images-500-{:s}-X.txt".format(tag)), "wb")
    fyout = open(os.path.join(data_dir, 
        "images-500-{:s}-y.txt".format(tag)), "wb")
    num_written = 0
    for i in range(0, X.shape[0], batch_size):
        Xbatch = X[i:i + batch_size]
        ybatch = y[i:i + batch_size]
        vecs = model.predict(Xbatch)
        for vec, label in zip(vecs, ybatch):
            vec = vec.flatten()
            vec_str = ",".join(["{:.5f}".format(v) for v in vec.tolist()])
            fXout.write("{:s}\n".format(vec_str))
            fyout.write("{:d}\n".format(label))
            if num_written % 100 == 0:
                print("\twrote {:d} {:s} records".format(num_written, tag))
            num_written += 1
    print("\twrote {:d} {:s} records, COMPLETE".format(num_written, tag))
    fXout.close()
    fyout.close()


########################## main ##########################

DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "images-500")
IMAGE_WIDTH = 224
BATCH_SIZE = 10
NUM_TO_AUGMENT = 10

np.random.seed(42)

# load images and labels from images directory
print("Loading images and labels from images directory...")
xs, ys = [], []
for label, image_file in get_next_image_loc(IMAGE_DIR):
    ys.append(int(label))
    img = image.load_img(os.path.join(IMAGE_DIR, label, image_file),
                         target_size=(IMAGE_WIDTH, IMAGE_WIDTH))
    img4d = image.img_to_array(img)
    img4d = np.expand_dims(img4d, axis=0)
    img4d = preprocess_input(img4d)
    xs.append(img4d[0])
X = np.array(xs)
y = np.array(ys)

# using regular train_test_split results in classes not being represented
print("Initial split into train/val/test...")
splitter = StratifiedShuffleSplit(y, n_iter=1, test_size=0.3, 
                                  random_state=42)
for train, test in splitter:
    Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
    break
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# instantiate ImageDataGenerator to create approximately 10 images for
# each input training image
print("Augmenting training set images...")
datagen = image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)

xtas, ytas = [], []
for i in range(Xtrain.shape[0]):
    num_aug = 0
    x = Xtrain[i][np.newaxis]
    datagen.fit(x)
    for x_aug in datagen.flow(x, batch_size=1):
       if num_aug >= NUM_TO_AUGMENT:
           break
       xtas.append(x_aug[0])
       ytas.append(ytrain[i])
       num_aug += 1

Xtrain = np.array(xtas)
ytrain = np.array(ytas)

print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# Instantiate VGG16 model and remove bottleneck
print("Instantiating VGG16 model and removing top layers...")
vgg16_model = VGG16(weights="imagenet", include_top=True)
model = Model(input=vgg16_model.input, 
              output=vgg16_model.get_layer("block5_pool").output)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy")

# Read each of train, validation and test vectors out to named files
print("Writing vectors to files...")
write_vectors(model, Xtrain, ytrain, "train", DATA_DIR, BATCH_SIZE)
write_vectors(model, Xtest, ytest, "test", DATA_DIR, BATCH_SIZE)
