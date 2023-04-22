import matplotlib
from matplotlib import pyplot as plt

import os

import cv2 as cv

import numpy as np

# import silence_tensorflow.auto # silence unnecessary warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_data(data_dir, image_dims):
        images = []
        labels = [] # buttons = 1, not buttons = 0
        for label in os.listdir(data_dir):
                if label == "button":
                        for image in os.listdir(data_dir + "/" + label):
                                image = cv.imread(data_dir + "/" + label + "/" + image)
                                image - cv.cvtColor(image, cv.COLOR_BGR2RGB)
                                image = cv.resize(image, image_dims)
                                images.append(image)
                                labels.append(1)
                elif label == "not_button":
                        for image in os.listdir(data_dir + "/" + label):
                                image = cv.imread(data_dir + "/" + label + "/" + image)
                                image - cv.cvtColor(image, cv.COLOR_BGR2RGB)
                                image = cv.resize(image, image_dims)
                                images.append(image)
                                labels.append(0)
                
        images = np.array(images, dtype="float32")
        labels = np.array(labels, dtype="int32")
        return images, labels

def main():
        # loading data
        images, labels = get_data("data", (128, 128))

        # TODO: delete
        print("Found " + str(len(images)) + " images of " + str(max(labels) + 1) + " classes ")

        # TODO: Consider Data augmentation

        # creating the model
        # The layers are as follows:
        # Convolution -> Max Pooling -> Convolution -> Max Pooling ->
        # Dense -> Output 
        model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation='sigmoid') # output layer; must be sigmoid
        ])

        # compiling the model
        model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=.001),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=keras.metrics.BinaryAccuracy()
        )

        # training model on data
        model.fit(
                images,
                labels,
                batch_size=30,
                epochs=25, 
                verbose=0, 
                validation_split=.2,
                shuffle=False, 
                validation_freq=5,
        )

        # saving model
        current_dir = os.getcwd()
        model.save(current_dir + "/model")


main()
