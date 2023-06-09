ElevatorButtonFinder
====================

Henry Nitzberg, 4/24/2023

Purpose and Overview:
---------------------
        This program is designed to find the elevator buttons in a given image 
        of the outside of an elevator. It uses a convolutional neural network 
        for binary classification in combination with select search to localize
        the regions of interest. It then uses k-means clustering to get a more 
        precise location of the button.

Usage:
------
        The program can be run from the command line, then reads from 
        standard input:
                python find_buttons.py OR
                path/to/image.jpg | python find_buttons.py

        The classifier can be trained from the command line, but the
        repository comes with a trained model, so this is not required:
                'python train_bin_class.py'
        train_bin_class.py requires that a directory named 'data' exists in the same
        directory as the script, and that it contains two subdirectories,
        'button' and 'not_button' with corresponding training images.
        
        Required Libraries:
                - numpy
                - tensorflow
                - opencv-python
                - keras (tensorflow.keras)
                - matplotlib
                - os
        Installing Required Libraries with Anaconda
                - enter base conda environment
                - run 'conda env create -n elevatorEnv -f environment.yml'
                - run 'conda activate elevatorEnv'
                - to ensure it worked simply run 'python find_buttons.py'

Modules:
--------
        train_bin_class.py
                This module is used to train the binary classifier (CNN) for the 
                buttons. The current model is trained on a fairly small dataset
                of ~150 images, augmented up to 1000.
                Classification accuracy of training and validation is nearly 
                100% with low loss, however in practice model yields many false 
                positives. 
        findButtons.py
                This module is used to find the buttons in an image. It uses 
                select search to find regions of interest, then uses the binary
                classifier to determine if the region is a button or not.
                It then displays the image with points plotted on the predicted
                buttons.

Credit:
-------
https://medium.com/analytics-vidhya/data-augmentation-techniques-using-opencv-657bcb9cc30b

https://realpython.com/k-means-clustering-python/

https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

https://keras.io/api/models/model_training_apis/

https://learnopencv.com/selective-search-for-object-detection-cpp-python/
