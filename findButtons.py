import matplotlib
from matplotlib import pyplot as plt

import os

import cv2 as cv

import numpy as np

import silence_tensorflow.auto # silence unnecessary warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_image_from_filepath():
        image = input("File Path to Image: ")
        image = cv.imread(image)

        if image is None:
                print("Invalid File Path")
                exit()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (250, int(image.shape[1]*250/image.shape[0])))
        return image

def main():
        # Note on Usage: ./data/... will get image from data folder
        # get image from command line
        image = get_image_from_filepath()

        selective_search = cv.ximgproc.segmentation.createSelectiveSearchSegmentation() # thanks again, google
        selective_search.setBaseImage(image)
        # selective_search.switchToSelectiveSearchFast()
        selective_search.switchToSelectiveSearchQuality() # TODO: try both and examine results

        recommended_boxes = selective_search.process()
        # shape: (x, y, w, h)
        to_check = []
        for x, y, w, h in recommended_boxes:
                box = image[y:y+h, x:x+w]
                box = cv.resize(box, (128, 128))
                to_check.append(box)

        to_check = np.array(to_check, dtype="float32")

        curr_path = os.getcwd()
        model = keras.models.load_model(curr_path + "/model")
        predictions = model.predict(to_check)

        # max prediction should be the button
        pred_list = predictions.tolist()
        max_predict = max(pred_list)
        print(max_predict)
        max_pred_index = pred_list.index(max_predict)
        x, y, w, h = recommended_boxes[max_pred_index]
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
        plt.imshow(image)
        plt.show()


main()
