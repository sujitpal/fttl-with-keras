# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_DIR = "../data/files/sample"

##################################
# Presentation Goals slide
##################################
def presentation_goals():
    files = os.listdir(IMAGE_DIR)
    fig, axes = plt.subplots(3, 3)
    axes = np.ravel(axes)
    for i in range(9):
        label = np.random.randint(4)
        files = os.listdir(os.path.join(IMAGE_DIR, str(label)))
        img_file = files[np.random.randint(len(files))]
        img = plt.imread(os.path.join(IMAGE_DIR, str(label), img_file))
        axes[i].imshow(img, interpolation="nearest")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

##################################
# What is DR slide
##################################
def what_is_dr():
    plt.subplot(511)
    img = plt.imread(os.path.join(IMAGE_DIR, "0", "13363_left.jpeg"))
    plt.title("No DR")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(512)
    img = plt.imread(os.path.join(IMAGE_DIR, "1", "14664_left.jpeg"))
    plt.title("Mild DR")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(513)
    img = plt.imread(os.path.join(IMAGE_DIR, "2", "14323_left.jpeg"))
    plt.title("Moderate DR")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(514)
    img = plt.imread(os.path.join(IMAGE_DIR, "3", "12612_right.jpeg"))
    plt.title("Severe DR")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(515)
    plt.title("Proliferative DR")
    img = plt.imread(os.path.join(IMAGE_DIR, "4", "15376_left.jpeg"))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()
    
######################## main ########################
#presentation_goals()
what_is_dr()