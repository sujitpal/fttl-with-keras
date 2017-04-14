# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import os

DOWNLOAD_DIR = "../data/files"
SHELL_SCRIPT = os.path.join(DOWNLOAD_DIR, "scriptImages.sh")
LABEL_FILE = os.path.join(DOWNLOAD_DIR, "trainLabels.csv")

label2images = {}
flab = open(LABEL_FILE, "rb")
for line in flab:
    if line.startswith("image,"):
        continue
    image_name, label = line.strip().split(",")
    if label2images.has_key(label):
        label2images[label] += image_name
    else:
        label2images[label] = [image_name]
flab.close()

fsh = open(SHELL_SCRIPT, "wb")
for label in label2images.keys():
    indices = np.arange(len(label2images[label]))
    print("label=", label, "len=", len(label2images[label]))
    sample_indices = np.random.choice(indices, size=200, replace=False)
    images = label2images[label]
    for ind in sample_indices:
        print("cp {:s}.jpeg sample/{:s}/".format(image_name, label))
        fsh.write("cp {:s}.jpeg sample/{:s}/\n".format(image_name, label))

fsh.close()
