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
        image = cv.resize(image, (int(image.shape[1]*250/image.shape[0]), 250))
        return image

def main():
        # Note on Usage: ./data/... will get image from data folder
        # get image from command line
        image = get_image_from_filepath()

        selective_search = cv.ximgproc.segmentation.createSelectiveSearchSegmentation() # thanks again, google
        selective_search.setBaseImage(image)

         #quality seems to outperform fast search
        selective_search.switchToSelectiveSearchQuality()

        recommended_boxes = selective_search.process()
        # shape: (x, y, w, h)

         # adding recommended regions as images to array
        to_check = []
        to_check_coords = []
        for x, y, w, h in recommended_boxes:
                # restrict to reasonable sizes and dimensions 
                if int(w / h * 10) in range(6, 14) \
                and w < image.shape[1] / 2 and h < image.shape[0] / 2 \
                and w > image.shape[1] / 10 and h > image.shape[0] / 10:
                        box = image[y:y+h, x:x+w]
                        box = cv.resize(box, (128, 128))
                        to_check.append(box)
                        to_check_coords.append([x, y, w, h])

        to_check = np.array(to_check, dtype="float32")

        # loading model
        curr_path = os.getcwd()
        model = keras.models.load_model(curr_path + "/model")
        predictions = model.predict(to_check)

        # max prediction should be the button
        pred_list = predictions.tolist()
        print("Num Predictions: " + str(len(pred_list)))

        # formating
        pred_list = [pred[0] for pred in pred_list]

        num_buttons = 0
        center_points = []
        for pred in pred_list:
                if pred >= .9:
                        x, y, w, h = to_check_coords[pred_list.index(pred)]

                        # rectangle around button guesses - good for debugging
                        # cv.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1) 

                        # center point of button guesses - used for kmeans below
                        center_points.append([x + w/2, y + h/2])
                        num_buttons += 1

        center_points = np.array(center_points, dtype="float32")

        if center_points == []:
                print("No Buttons Found")
                exit()
        elif num_buttons == 1:
                print("Only 1 Button Found")
                cv.circle(image, (int(center_points[0][0]), int(center_points[0][1])), 5, (0, 0, 255), -1)
                plt.imshow(image)
                plt.show()
                exit()

        # credit: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,centers=cv.kmeans(center_points,2,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

        print("predicted regions: " + str(num_buttons))
        for x, y in centers:
                cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        plt.imshow(image)
        plt.show()

main()
