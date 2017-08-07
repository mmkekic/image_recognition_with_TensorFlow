# tensorflow-exercise
simple CNN for image classification
The code serves for image classification of three Barcelona football players, data provided as exercise by Methinks software company.
dataset.py contains DataSet class that serves to read divide dataset in train, test and validation data and make batches of train data. It also reads labeling of the data given in the file name.
train.py builds CNN and train it on the data, as well as testing it preformance on validation data.
