import tensorflow as tf
import matplotlib as plt
import os

from tensorflow import keras


def main():
    # generating a dataset using keras
    training_images, testing_images = tf.keras.utils.image_dataset_from_directory(
        "data",
        labels='inferred', # data is already sorted within the data directory
                           # so keras labels it
        validation_split=.2,
        subset="both", # return both training images and testing images
        image_size=(128, 128),
        batch_size=30
    )


main()
