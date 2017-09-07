# image recognition with TensorFlow

This is a simple classification of three Barcelona football players (Messi, Iniesta and Neymar;  data provided as exercise by Methinks software company).

## Prerequisites
The code uses [TensorFlow] (https://www.tensorflow.org/) for implementing and training Neural Network. Images handling is done with scipy and skimage library.

## Task description
The goal is to build and train neural network (NN) that will serve to classify photos of the three football players. The labeling of the photos is given in the image captions. Resolution of all images is 60x60

## Code structure

The code consists of dataset.py module that prepares dataset for training and testing and train.py main file that builds, train and test a NN.

## Data handling
Dataset is divided into train (70%), test (30%) and validation (20%)  data.
Functions test_image and valid_images are returning corresponding sets of labeled images, while function next_batch(batch_size) is returning a batch of train images set.
In order to avoid overfitting the train set is augmented with random rotations, translations, zoom or light, done in image_proc function.
The images are normalized to -1:1 and 0 centered (by subtracting the mean).

## Neural Network architecture
The network is a convolutional neural network consisting of following layers:
*INPUT(60x60x3)
*CONVOLUTIONAL (5x5x3, 32)
*MAXPOOL (2x2)
*CONVOLUTIONAL (3x3x32, 64)
*MAXPOOL (2x2)
*CONVOLUTIONAL (3x3x64, 128)
*MAXPOOL (3x3)
*FULLY CONNECTED (7x7x128, 1024)
*OUT (FULLY CONNECTED) (1024, 3)

Overfitting is avoided using dropout, L2 regularization and early stopping (when error on validation data gets to its minimum).

## Running the code

After installing necessary libraries and putting data inside ./fcbdata/ folder of working directory, running 'train' without parameters will build and train the network, while running 'train test' will try to load previously trained network and test the accuracy on the test data set.


## For information about NN I strongly recommend Andrej Karpathy online course http://cs231n.stanford.edu/
