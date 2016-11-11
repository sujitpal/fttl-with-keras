# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import numpy as np
import os

import fttlutils

##################### main ######################

DATA_DIR = "../data/files"
MODEL_DIR = os.path.join(DATA_DIR, "models")

# data
X = np.loadtxt(os.path.join(DATA_DIR, "images-X.txt"), delimiter=",")
y = np.loadtxt(os.path.join(DATA_DIR, "images-y.txt"), delimiter=",", 
               dtype=np.int)

Xtrain, Xtest, ytrain, ytest = fttlutils.train_test_split(
    X, y, test_size=0.3, random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# model
clf = LogisticRegression()
clf.fit(Xtrain, ytrain)

ytest_ = clf.predict(Xtest)
fttlutils.print_stats(ytest, ytest_, "LR Model")
with open(os.path.join(MODEL_DIR, "lr-model.pkl"), "wb") as fmodel:
    pickle.dump(clf, fmodel)
