import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
from skimage import transform
from skimage import exposure
from skimage import io


class DataSet(object):
    ''' The class contains all relevant information about data. Should be called with the data_path argument.'''
    def __init__(self,data_path,test_size=0.3,valid_size=0.2):
        self.data_path=data_path
        #get names of all the images
        self.names=np.array(os.listdir(data_path))
        self.names.sort()
        
        #shuffle names so we can split it in train-test-validation data,
        #fix random_seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(self.names)

        #define size of train, test, validation data and divide image set
        self.datasize=len(self.names)
        self.valid_size=int(valid_size*len(self.names))
        self.test_size=int(test_size*len(self.names))
        self.train_size=len(self.names)-self.valid_size-self.test_size

        #train/valid/test are lists of image filenames correspoiding to train/valid/test splitting
        self.train=self.names[:self.train_size]
        self.valid=self.names[self.train_size:self.train_size+self.valid_size]
        self.test=self.names[self.train_size+self.valid_size:]

        #define categories      
        self.cat=[ 'iniesta','neymar','messi']
        #one-hot-encoder dictionary, max_value index corresponds to self.cat
        self.cat_to_ohe={'iniesta':[1,0,0],'neymar':[0,1,0],'messi':[0,0,1]}
       
        #train/valid/test_labels are lists of labels correspoiding to train/valid/test splitting
        self.train_labels=np.array(map(self.get_cat,self.train))
        self.valid_labels=np.array(map(self.get_cat,self.valid))
        self.test_labels=np.array(map(self.get_cat,self.test))
        
        #variables that cout batches inside every epoch and how many epochs are complited
        self.batch_counts=0
        self.epoch_counts=0
        self.im_size=60

        #get mean values per channal for all training image (for data centralization)
        #since all the images are the same size mean value of all images is also mean value of their means
        means=np.zeros(3)
        for x in self.train:
            img = io.imread(self.data_path + x)
            img=skimage.img_as_float(img)
            img_mean=np.mean(img,axis=(0,1))
            means=means+img_mean
        self.mean=means/len(self.train)
        
    def next_batch(self,batch_size):
        '''The function gives next batch of training images. Input is batch size and output arrays of training images and correspoiding labels. 
        The shape of the output is X(batch_sizex60x60x3), y(batch_sizex3).
        '''
        start=self.batch_counts*batch_size
        #the last batch will be smaller if training set size not dividable by batch size
        end=min((self.batch_counts+1)*batch_size,self.train_size)
        batch_names=self.train[start:end]
        X_train = np.empty(shape=(len(batch_names),self.im_size, self.im_size,3))
        #read and transform images 
        count=0
        for x in batch_names:
            img = io.imread(self.data_path + x)
            img=skimage.img_as_float(img)
            img=image_proc(img)-self.mean
            X_train[count]=img
            count+=1
        
        #get image labels
        batch_labels=self.train_labels[start:end]
        y_train=np.array(map(lambda x:self.cat_to_ohe[x],batch_labels))

        self.batch_counts+=1
        #increase epoch_counts if all training set sampled, reset batch_counts and shuffle training set
        if (self.batch_counts)*batch_size > self.train_size:
           
            self.epoch_counts+=1
            self.batch_counts=0
            indx=np.arange(0,self.train_size)
            np.random.shuffle(indx)
            self.train=self.train[indx]    
            self.train_labels=self.train_labels[indx]
        return X_train,y_train


    def test_image(self):
        '''The function gives test images. The output is array of test images and correspoiding labels. 
        The shape of the output is X(test_sizex60x60x3), y(test_sizex3).
        '''
        X_test = np.empty(shape=(self.test_size,self.im_size, self.im_size,3))

        count=0
        for x in self.test:
            img = io.imread(self.data_path + x)
            img=skimage.img_as_float(img)-self.mean
            X_test[count]=img
            count+=1

        y_test=np.array(map(lambda x:self.cat_to_ohe[x],self.test_labels))
        return X_test,y_test


    def valid_image(self):
        '''The function gives validation images. The output is array of validation images and correspoiding labels. 
        The shape of the output is X(valid_sizex60x60x3), y(valid_sizex3).
        '''
        X_valid = np.empty(shape=(self.valid_size,self.im_size, self.im_size,3))
        count=0
        for x in self.valid:
            img = io.imread(self.data_path + x)
            img=skimage.img_as_float(img)-self.mean
            X_valid[count]=img
            count+=1

        y_valid=np.array(map(lambda x:self.cat_to_ohe[x],self.valid_labels))          
        return X_valid,y_valid


    def get_cat(self,string):
        '''returns category from a string'''
        string=string.lower()
        for i in self.cat:
            if i in string:
                return i
        return None

    
#image transformations
def image_proc(image):
    ''' This function
    performs random transformation on the image:
    random rotation between -5:5 degrees, 
    random translation between -10:10
    random zoom 1:1.5 in about 50% of the cases
    random gamma correction in range 0.8-1.2
    Input is image as numpy array and it outputs the transformed image '''

    size=image.shape[0]

    #random rotations betweein -5 and 5 degrees
    deg = np.random.uniform(-5,5)

    #random translations
    trans_1 = np.random.uniform(-10,10)
    trans_2 = np.random.uniform(-10,10)
    
    #random zooms
    zoom = np.random.uniform(0.8, 1.5)
    zoom=max(1.,zoom)

    #shearing
    shear_deg = np.random.uniform(-5, 5)

    #apply spatial transformations
    center_shift   = np.array((size, size)) / 2. - 0.5
    tform_center=transform.SimilarityTransform(translation=-center_shift)
    tform_aug=transform.AffineTransform(rotation = np.deg2rad(deg),
                                        scale =(1/zoom, 1/zoom),
                                        shear = np.deg2rad(shear_deg),
                                        translation = (trans_1, trans_2))
    tform_uncenter=transform.SimilarityTransform(translation=center_shift)
    tform=tform_center+ tform_aug+ tform_uncenter
    image=transform.warp(image,tform)

    #change gamma
    gamma=np.random.uniform(0.8,1.2)
    image=exposure.adjust_gamma(image,gamma)
    
    return image



