ElevatorButtonFinder
====================

Henry Nitzberg, 4/24/2023

Purpose and Overview:
---------------------
        <p>
        This program is designed to find the elevator buttons in a given image 
        of the outside of an elevator. It uses a convolutional neural network 
        for binary classification in combination with select search to localize
        the regions of interest. It then uses k-means clustering to get a more 
        precise location of the button.
        </p>

Usage:
------
        <p>
        The program can be run from the command line, then reads from 
        standard input:
                'python findButtons.py' OR
                'path/to/image.jpg | python findButtons.py'
        </p>
        <p>
        The binary classifier can also be trained from the command line, but the
        repository comes with a trained model, so this is not required:
                'python train_bin_class.py'
        However, it requires that a directory named 'data' exists in the same
        directory as the script, and that it contains two subdirectories,
        'button' and 'not_button' with corresponding training images.
        </p>
        <p>
        Required Libraries:
                - numpy
                - tensorflow
                - opencv-python
                - keras (tensorflow.keras)
                - matplotlib
                - os
        </p>

Modules:
--------
        <p>
        train_bin_class.py
                This module is used to train the binary classifier (CNN) for the 
                buttons. The current model is trained on a fairly small dataset
                of ~150 images, augmented up to 1000.
                Training and validation accuracy and is close to 100% but in
                practice the model yields many false positives. 
        </p>
        <p>
        findButtons.py
                This module is used to find the buttons in an image. It uses 
                select search to find regions of interest, then uses the binary
                classifier to determine if the region is a button or not.
                It then displays the image with points plotted on the predicted
                buttons.
        </p>

Credit:
-------
<p>
https://medium.com/analytics-vidhya/data-augmentation-techniques-using-opencv-
657bcb9cc30b

https://realpython.com/k-means-clustering-python/

https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
</p>
