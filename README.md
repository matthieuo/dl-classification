# dl-classification
This project allows you to create a multi-classes classifier for images with deep learning. The  [TensorFlow](http://www.tensorflow.org/) framework is used for the computations.
 

## Usage
This project reads data from directories organized as follow:


```
data_train/classe1/[jpeg images from classe_1]
          /classe2/[jpeg images from classe_2]
          .....
          /classen/[jpeg images from classe_n]
```


To create a classifier by training a deep neural network:

`train.py -paths "data_train"  -nc 3  -reg 0.0001  -dp .4 -s "training1" -bp "OUTPUT_PATH" -bs 64` 

In with `-paths` contains the training set, `-nc` controls the number of classes, `-reg` controls the L2 regularization factor, `-dp` controls the dropout value. 

A new directory is created on the `-bp` directory named with the training parameter plus the `-s` string. This directory contains the trained models and some metrics for tensorboard.


It's possible to test the trained models on a test set. The test set's directory should be organized like the train set directory.






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



