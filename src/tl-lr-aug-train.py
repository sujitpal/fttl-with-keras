# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import cPickle as pickle
import numpy as np
import os

DATA_DIR = "../data"

# data
print("Loading data...")
Xtrain = np.loadtxt(os.path.join(DATA_DIR, "images-500-train-X.txt"), 
                    delimiter=",")
ytrain = np.loadtxt(os.path.join(DATA_DIR, "images-500-train-y.txt"), 
                    delimiter=",", dtype=np.int)
print("\ttrain:", Xtrain.shape, ytrain.shape)

Xtest = np.loadtxt(os.path.join(DATA_DIR, "images-500-test-X.txt"), 
                   delimiter=",")
ytest = np.loadtxt(os.path.join(DATA_DIR, "images-500-test-y.txt"), 
                   delimiter=",", dtype=np.int)
print("\ttest:", Xtest.shape, ytest.shape)
            
np.random.seed(42)

# model
clf = LogisticRegression()
clf.fit(Xtrain, ytrain)

ytest_ = clf.predict(Xtest)

print("Accuracy: {:.3f}".format(accuracy_score(ytest, ytest_)))
print("Confusion Matrix:")
print(confusion_matrix(ytest, ytest_))
print("Classification Report:")
print(classification_report(ytest, ytest_))

with open(os.path.join(DATA_DIR, "lr-model-aug.pkl"), "wb") as fmodel:
    pickle.dump(clf, fmodel)
