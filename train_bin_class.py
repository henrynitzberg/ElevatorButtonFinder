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


def get_data(data_dir, image_dims):
        images = []
        labels = [] # buttons = 1, not buttons = 0
        for label in os.listdir(data_dir):
                for image in os.listdir(data_dir + "/" + label):
                        image = cv.imread(data_dir + "/" + label + "/" + image)
                        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        image = cv.resize(image, image_dims)
                        images.append(image)
                labels.append(1) if label == "button" else labels.append(0)
                        
        images = np.array(images, dtype="float32")
        labels = np.array(labels, dtype="int32")
                
        return images, labels


def augment_image(image):
        # random rotation
        angle = random.randint(-20, 20)
        matrix = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
        augmented_image = cv.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        # random flip (0 = vertical, 1 = horizontal, -1 = both)
        augmented_image = image
        flip = random.choice([0, 1, -1, None])
        if flip is not None:
                augmented_image = cv.flip(image, flip)

        # random brightness
        brightness = random.randint(-50, 50)
        augmented_image = cv.addWeighted(augmented_image, 1, np.zeros(augmented_image.shape, augmented_image.dtype), 0, brightness) #thanks AI!

        # random contrast
        contrast = random.randint(-50, 50)
        augmented_image = cv.addWeighted(augmented_image, 1 + contrast / 100, np.zeros(augmented_image.shape, augmented_image.dtype), 0, 0) #thanks AI!

        cv.resize(augmented_image, (128, 128))
        return augmented_image


def main():
        # loading data
        images, labels = get_data("data_bin", (128, 128))
        print("Found " + str(len(images)) + " images of " + str(max(labels) + 1) + " classes ")

        # data augmentation (~50 samples -> 1000 samples)
        goalDataSize = 1000
        print("Augmenting data to " + str(goalDataSize) + " samples...")
        originalDataSize = len(images)
        while len(images) < goalDataSize:
                for i in range(originalDataSize):
                        images = np.append(images, [augment_image(images[i])], axis=0)
                        labels = np.append(labels, [labels[i]], axis=0)
        print("Data augmentation complete")

        # plt.imshow(images[0].astype(np.uint8))
        # plt.show()

        # plt.imshow(images[originalDataSize].astype(np.uint8))
        # plt.show()

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
                optimizer=keras.optimizers.Adam(learning_rate=.0005),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=keras.metrics.BinaryAccuracy()
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

        # saving model
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
        


main()
