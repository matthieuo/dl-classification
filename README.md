


# dl-classification

[![Build Status](https://travis-ci.org/matthieuo/dl-classification.svg?branch=master)](https://travis-ci.org/matthieuo/dl-classification)


This project allows you to create a multiclass classifier for images with deep learning. The  [TensorFlow](http://www.tensorflow.org/) framework is used for the computations.
 
## Installation
The necessary Python packages are listed on [requirments.txt](requirments.txt). Training a deep neural network is done with the `train.py` file.


## Usage
This project reads data from directories organized as follow:


```
data_train/class_1/[jpeg images from class_1]
          /class_2/[jpeg images from class_2]
          .....
          /class_n/[jpeg images from class_n]
```


To create a classifier by training a deep neural network:

`train.py -paths "data_train"  -nc 3  -reg 0.0001  -dp .4 -s "training1" -bp "OUTPUT_PATH" -bs 64` 

In which `-paths` contains the training set, `-nc` controls the number of classes, `-reg` controls the L2 regularization factor, `-dp` controls the dropout value. 


*Note:* the number of classes set with the  `-nc` argument and the number of classes on the training set must be strictly identical.

A new directory is created on the `-bp` directory named with the training parameter plus the `-s` string. This directory contains the trained models and some metrics for tensorboard.


It's possible to test the trained models on a test set with the `-tp` option. The test set's directory should be organized like the train set directory.



## Code organization

* load_images.py
    * Initialization of the queues to load images and perform data augmentation
* train.py
    * Train models for binary classification
* eval.py
    * Evaluation of trained models
* inference.py
    * Simple class to perform inference
* initializers.py
    * Helper functions to initialize networks
* models.py
    * Implementation of the DL model



