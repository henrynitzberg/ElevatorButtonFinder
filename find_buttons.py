# 
# findButtons.py
# Henry Nitzberg
# 4/24/2023
#
# This program uses a pretrained model to find buttons in an image
# It uses selective search to find regions of interest, and then
# uses the model to predict whether or not the region is a button
# It outputs the origional image with blue circles on the predicted buttons
#

import matplotlib
from matplotlib import pyplot as plt

import os

import cv2 as cv

import numpy as np

import silence_tensorflow.auto # silence unnecessary warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# get_image_from_filepath
# Args: This function takes no arguments
# Returns: This function returns a 3D numpy array representing an image
# Notes: This function prompts the user for a file path to an image
#        It then reads the image, converts it to RGB, and resizes it
#        so the height is 250 pixels; width is scaled accordingly
def get_image_from_filepath():
        image = input("File Path to Image: ")
        image = cv.imread(image)

        if image is None:
                print("Invalid File Path")
                exit()

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (int(image.shape[1]*250/image.shape[0]), 250))
        return image

# get_model
# Args: an image, and an array of coordinates of regions of interest
# Returns: This function returns a tuple of two numpy arrays, the first is
#          an array of images, the second is an array of the coordinates of the
#          regions of interest
# Notes: This functions narrows down the predictions with some common sense - 
#        it only checks regions of interest that are a reasonable size and shape
#        It also resizes the regions of interest to 128x128, the same size as
#        the images the model was trained on
def get_recommendations(image, recommended_boxes):
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
        return to_check, to_check_coords

# get_centers
# Args: an array of predictions, and an array of coordinates of regions of 
#       interest
# Returns: This function returns a tuple of two values, the first is the number
#          of buttons predicted, the second is an array of the centers of the
#          predicted buttons
# Notes: This function gets the centers of the predicted buttons, if the
#        prediction is above a certain threshold
def get_centers(predictions, to_check_coords):
        num_buttons = 0
        center_points = []
        for pred in predictions:
                if pred >= .9:
                        x, y, w, h = to_check_coords[predictions.index(pred)]
                        # rectangle around button guesses - good for debugging
                        # model + select search
                        # cv.rectangle(image, 
                        #              (x, y), 
                        #              (x+w, y+h), 
                        #              (255, 0, 0), 
                        #              1,
                        #              ) 

                        # center point of button guesses - used for kmeans below
                        center_points.append([x + w/2, y + h/2])
                        num_buttons += 1

        center_points = np.array(center_points, dtype="float32")

        return num_buttons, center_points

# get_precise_centers
# Args: an array of centers of predicted buttons
# Returns: This function returns an array of the centers of the predicted
#          buttons, after running kmeans on them (with 2 classes, to represent
#          the two buttons; up, and down)
def get_precise_centers(center_points):
        # credit: 
        # https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _,_,centers=cv.kmeans(center_points,
                              2,
                              None,
                              criteria,
                              10,
                              cv.KMEANS_RANDOM_CENTERS
                             )
        return centers

# main method
def main():
        # get image from command line
        image = get_image_from_filepath()

        # get recommended regions of interest
        selective_search = \
                cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        selective_search.setBaseImage(image)

         #quality seems to outperform fast search
        selective_search.switchToSelectiveSearchQuality()

        recommended_boxes = selective_search.process()
        # shape: [[x, y, w, h], ...]

        # getting recommended regions as images in array, and their coordinates
        to_check, to_check_coords = get_recommendations(image, 
                                                        recommended_boxes
                                                       )

        # loading model
        curr_path = os.getcwd()
        model = keras.models.load_model(curr_path + "/model")

        # classifying each region of interest
        predictions = model.predict(to_check)

        predictions = predictions.tolist()
        print("Num Predictions: " + str(len(predictions)))

        # formatting
        predictions = [pred[0] for pred in predictions]

        # finding centers of predicted buttons
        num_buttons, center_points = get_centers(predictions, to_check_coords)

        if center_points == []:
                print("No Buttons Found")
                exit()
        elif num_buttons == 1:
                print("Only 1 Button Found")
                cv.circle(image, 
                          (int(center_points[0][0]), int(center_points[0][1])), 
                          5, 
                          (0, 0, 255), 
                          -1,
                         )
                plt.imshow(image)
                plt.show()
                exit()

        # kmeans clustering to find more precise location of buttons
        centers = get_precise_centers(center_points)

        print("predicted regions: " + str(num_buttons))
        for x, y in centers:
                cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        plt.imshow(image)
        plt.show()

main()
