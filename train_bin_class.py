# 
# train_bin_class.py
# Henry Nitzberg
# 4/24/2023
#
# This program trains a CNN model to classify images as buttons or not buttons
# It uses data augmentation to increase the size of the dataset to 1000.
# It outputs the average loss and accuracy of the model, then provides the
# client with the option to save the model into the current directory as 
# ./model. It is loaded in findButtons.py
#

import matplotlib
from matplotlib import pyplot as plt

import os

import cv2 as cv

import random

import numpy as np

import silence_tensorflow.auto # silence unnecessary warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# get_data
# Args: This function takes a directory and desired image dimensions
# Returns: This function returns a tuple of two numpy arrays, the first is
#          an array of images, the second is an array of labels
# Notes: This functions assumes that the data directory given is organized into
#        two sub-directories, "button" and "not_button", and that the images
#        are in the .jpg format
def get_data(data_dir, image_dims):
        images = []
        labels = [] # button = 1, not button = 0
        for label in os.listdir(data_dir):
                for image in os.listdir(data_dir + "/" + label):
                        image = cv.imread(data_dir + "/" + label + "/" + image)
                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        image = cv.resize(image, image_dims)
                        images.append(image)
                        if label == "button":
                                labels.append(1)
                        else:
                                labels.append(0)
                        
        images = np.array(images, dtype="float32")
        labels = np.array(labels, dtype="int32")

        print("Found " + str(len(images)) + " images of " + 
              str(max(labels) + 1) + " classes")
                
        return images, labels

# augment_image
# Args: This function takes an image (3D numpy array)
# Returns: This function returns an augmented image (3D numpy array)
# Notes: This function randomly rotates, flips, and adjusts the brightness and
#        contrast of the image.
def augment_image(image):
        # random rotation
        angle = random.randint(-20, 20)
        matrix = cv.getRotationMatrix2D(
                                        (image.shape[1] / 2, 
                                         image.shape[0] / 2), 
                                        angle, 
                                        1,
                                       )
        
        augmented_image = cv.warpAffine(image, matrix, 
                                        (image.shape[1], 
                                         image.shape[0])
                                       )

        # random flip (0 = vertical, 1 = horizontal, -1 = both)
        augmented_image = image
        flip = random.choice([0, 1, -1, None])
        if flip is not None:
                augmented_image = cv.flip(image, flip)

        # random brightness
        brightness = random.randint(-50, 50)
        augmented_image = cv.addWeighted(augmented_image, 
                                         1, 
                                         np.zeros(augmented_image.shape, 
                                                  augmented_image.dtype), 
                                         0, 
                                         brightness
                                        )

        # random contrast
        contrast = random.randint(-50, 50)
        augmented_image = cv.addWeighted(augmented_image, 
                                         1 + contrast / 100, 
                                         np.zeros(augmented_image.shape, 
                                                  augmented_image.dtype), 
                                         0, 
                                         0)

        cv.resize(augmented_image, (128, 128))
        return augmented_image

# save_model_querey
# Args: This function takes a model
# Returns: This function returns nothing
# Notes: This function asks the client if they want to save the model, and if
#        they do, it saves the model as ./model
def save_model_querey(model):
        print("Save Model? (y/n)")
        while (True):
                answer = input()
                if answer == "y":
                        current_dir = os.getcwd()
                        model.save(current_dir + "/model")
                        print("Model saved as " + current_dir + "/model")
                        break
                elif answer == "n":
                        print("Model not saved")
                        break
                else:
                        print("Invalid input")

# augment_data
# Args: This function takes an array of images, an array of labels, and a goal
#       data size
# Returns: This function returns nothing
# Notes: This function augments the data until the number of samples is greater
#        than or equal to the goal data size
def augment_data(images, labels, goalDataSize):
        print("Augmenting data to " + str(goalDataSize) + " samples...")
        originalDataSize = len(images)
        while len(images) < goalDataSize:
                for i in range(originalDataSize):
                        images = np.append(images, 
                                           [augment_image(images[i])], 
                                           axis=0
                                          )
                        labels = np.append(labels, [labels[i]], axis=0)
        print("Data augmentation complete")

# main method
def main():
        # loading data
        images, labels = get_data("data_bin", (128, 128))

        # data augmentation (~50 samples -> 1000 samples)
        goalDataSize = 1000
        augment_data(images, labels, goalDataSize)
       
        # creating the model
        # The layers are as follows:
        # Convolution -> Max Pooling -> Convolution -> Max Pooling ->
        # Dense -> Output 
        model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=(128, 128, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                # Note: must be sigmoid for binary classification
                layers.Dense(1, activation='sigmoid'),
        ])

        # compiling the model
        model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=.0005),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=keras.metrics.BinaryAccuracy(),
        )

        # training model on data
        print("Training Model...")
        model.fit(
                images,
                labels,
                batch_size=32,
                epochs=25, 
                verbose=2, 
                validation_split=.2,
                shuffle=False, 
                validation_freq=5,
        )


        # print accuracy and loss
        print("Accuracy: " + str(model.evaluate(images, labels, verbose=0)[1]))
        print("Loss: " + str(model.evaluate(images, labels, verbose=0)[0]))

        # save model
        save_model_querey(model)

main()
