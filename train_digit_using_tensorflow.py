# USAGE
# python train_digit_using_tensorflow.py --model output/digit_classifier.h5

# import the necessary packages
from pyimagesearch.models import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from skimage.segmentation import clear_border
import numpy as np
import argparse
import os
import cv2

# construct the argument parser and parse the arguments
'''ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model after training")
args = vars(ap.parse_args())'''

# simplify args:
path = "output/digit_classifier.h5"
args = {"model": path}

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 2
BS = 128

# grab the MNIST dataset
print("[INFO] accessing MNIST...")
((train_numeric_data, train_numeric_labels), (test_numeric_data, test_numeric_labels)) = mnist.load_data()

# add a channel (i.e., grayscale) dimension to the digits
train_numeric_data = train_numeric_data.reshape((train_numeric_data.shape[0], 28, 28, 1))
test_numeric_data = test_numeric_data.reshape((test_numeric_data.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
train_numeric_data = train_numeric_data.astype("float32") / 255.0
test_numeric_data = test_numeric_data.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
train_numeric_labels = le.fit_transform(train_numeric_labels)
test_numeric_labels = le.transform(test_numeric_labels)
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
if os.path.exists(args["model"]):
    model = load_model(args["model"])
else:
    model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

# train the network (pending)
print("[INFO] training numeric network...")
H = model.fit(
    train_numeric_data, train_numeric_labels,
    validation_data=(test_numeric_data, test_numeric_labels),
    batch_size=BS,
    epochs=EPOCHS,
    verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(test_numeric_data)
print(classification_report(
    test_numeric_labels.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]))

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")
