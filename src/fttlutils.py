# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

def train_test_split(X, Y, test_size, random_state):
    # using regular train_test_split results in classes not being represented
    splitter = StratifiedShuffleSplit(n_splits=1, 
                                      test_size=test_size, 
                                      random_state=random_state)
    for train, test in splitter.split(X, Y):
        Xtrain, Xtest, Ytrain, Ytest = X[train], X[test], Y[train], Y[test]
        break
    return Xtrain, Xtest, Ytrain, Ytest
    
def plot_loss(history):
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
    
def print_stats(ytest, ytest_, model_name):
    print(model_name)
    print("Accuracy: {:.5f}, Cohen's Kappa Score: {:.5f}".format(
        accuracy_score(ytest, ytest_), 
        cohen_kappa_score(ytest, ytest_, weights="quadratic")))
    print("Confusion Matrix:")
    print(confusion_matrix(ytest, ytest_))
    print("Classification Report:")
    print(classification_report(ytest, ytest_))

