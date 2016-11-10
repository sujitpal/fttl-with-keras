# Data for this project

Data for this project comes from the [Kaggle Diabetic Retinopathy Detection Competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). We use a random sample of 1,000 images from the 35,126 images in this dataset.

The following Makefile will create the directory structure under this directory. Unfortunately, the curl commands download HTML for the train.00[1-5].zip files. You will need to download them off the site via the browser.

You will need to make the credentials.txt file using the provided template file. Replace the values with your Kaggle user name and password. Run the following commands.

    make
    cd files

Next manually download trainLabels.csv.zip and sampleSubmission.csv.zip onto the HTML versions the make command downloaded. Also manually download the train.zip.00[1-5] files over the HTML versions downloaded.

    unzip -a trainLabels.csv.zip
    rm trainLabels.csv.zip
    unzip -a sampleSubmission.csv.zip
    rm sampleSubmission.csv.zip
    cd files
    rm test.zip*
    7za x train.zip.001

The last command will write 35126 images into the train directory under the current directory (files).

You can now build the sample from the dataset.

    cd ..
    ../../src/make-sample.py

This will generate a shell script sampleImages.sh in current directory.

    cd files
    mkdir sample
    cd sample
    mkdir 0 1 2 3 4
    cd ../..
    bash sampleImages.sh

This will copy 1000 images, 200 each into each of the 5 category directories.

Finally create a directory to hold the models.

    cd .. # you are in files now
    mkdir models

