# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.utils import np_utils
from sklearn.metrics import *
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "../data"
NUM_EPOCHS = 75
BATCH_SIZE = 32

# data
print("Loading data...")
Xtrain = np.loadtxt(os.path.join(DATA_DIR, "images-500-train-X.txt"), 
                    delimiter=",")
ytrain = np.loadtxt(os.path.join(DATA_DIR, "images-500-train-y.txt"), 
                    delimiter=",", dtype=np.int)
Ytrain = np_utils.to_categorical(ytrain - 1, nb_classes=5)
print("\ttrain:", Xtrain.shape, Ytrain.shape)

Xtest = np.loadtxt(os.path.join(DATA_DIR, "images-500-test-X.txt"), 
                   delimiter=",")
ytest = np.loadtxt(os.path.join(DATA_DIR, "images-500-test-y.txt"), 
                   delimiter=",", dtype=np.int)
Ytest = np_utils.to_categorical(ytest - 1, nb_classes=5)        
print("\ttest:", Xtest.shape, Ytest.shape)
            
np.random.seed(42)

# model
# input: (None, 25088)
imgvecs = Input(shape=(Xtrain.shape[1],), dtype="float32")
# hidden layer: (None, 256)
fc1 = Dense(256, activation="relu")(imgvecs)
fc1_drop = Dropout(0.5)(fc1)
# output layer: (None, 5)
labels = Dense(5, activation="softmax")(fc1_drop)

## model 2
## input: (None, 25088)
#imgvecs = Input(shape=(Xtrain.shape[1],), dtype="float32")
## hidden layer: (None, 2048)
#fc1 = Dense(4096, activation="relu")(imgvecs)
#fc1_drop = Dropout(0.5)(fc1)
## hidden layer: (None, 256)
#fc2 = Dense(256, activation="relu")(fc1_drop)
#fc2_drop = Dropout(0.5)(fc2)
## output layer: (None, 5)
#labels = Dense(5, activation="softmax")(fc2_drop)


model = Model(input=[imgvecs], output=[labels])

model.compile(optimizer="adadelta", loss="categorical_crossentropy",
              metrics=["accuracy"])

best_model = os.path.join(DATA_DIR, "dl-model-aug-best.h5")
checkpoint = ModelCheckpoint(filepath=best_model, verbose=1, 
                             save_best_only=True)
history = model.fit([Xtrain], [Ytrain], nb_epoch=NUM_EPOCHS,
                    batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks=[checkpoint])

# visualize training loss and accuracy
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="r", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="r", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# evaluate final model
Ytest_ = model.predict(Xtest)
ytest = np_utils.categorical_probas_to_classes(Ytest) + 1
ytest_ = np_utils.categorical_probas_to_classes(Ytest_) + 1

print("Final model")
print("Accuracy: {:.5f}".format(accuracy_score(ytest_, ytest)))
print("Confusion Matrix:")
print(confusion_matrix(ytest_, ytest))
print("Classification Report:")
print(classification_report(ytest_, ytest))

model.save(os.path.join(DATA_DIR, "dl-model-aug-final.h5"))

# load best model and evaluate

model = load_model(os.path.join(DATA_DIR, "dl-model-aug-best.h5"))
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
              metrics=["accuracy"])
Ytest_ = model.predict(Xtest)
ytest = np_utils.categorical_probas_to_classes(Ytest) + 1
ytest_ = np_utils.categorical_probas_to_classes(Ytest_) + 1

print("Best model")
print("Accuracy: {:.5f}".format(accuracy_score(ytest_, ytest)))
print("Confusion Matrix:")
print(confusion_matrix(ytest_, ytest))
print("Classification Report:")
print(classification_report(ytest_, ytest))

