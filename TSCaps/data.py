import TSCaps.config as cfg
import numpy as np
import os 
import matplotlib.pyplot as plt
from keras.utils import  np_utils
#
#
# reading data from the txt files
#
def readucr(filename):
    data = np.loadtxt(filename+".tsv", delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y
###
#    
#  To normalize the trained lebeled data
def NormalizationClassification(Y,num_classes):
    Y = np.array(Y)
    return (Y-Y.mean()) / (Y.max()-Y.mean()) *(num_classes-1)
#
#
#
def NormalizationFeatures(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X-mean) / std
#
#  This file is used to get the size of class in the training dataset
#
def GetNumClasses(y):
    y = np.array(y)
    num_classes = len(np.unique(y))
    return num_classes
#### Noising
#   Using the Gussian function 
#
def Noising(x,loc=cfg.loc,scale=cfg.scale):
    x = np.array(x)
    shape = x.shape
    x = np.random.normal( loc=loc,scale=scale,size=shape) + x
    return x
#
#   To OneHot
#
def OneHot(y,num_classes):
    y = np_utils.to_categorical(y,num_classes)
    return y
#
#
#Show The index of picure
#
def Show(train_x,aug_x,index,length):
    x = [i for i in range(1,length+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=2,ncols=1)
    aix[0].plot(x,train_x[index])
    aix[1].plot(x,aug_x[index])
    plt.show()
#
#
# Agumentation 
#
def Augmentation(train_x):
    x_shape = train_x.shape
    list_len = len(cfg.locslist)
    new_train = np.array(np.zeros((x_shape[0]*(list_len),x_shape[1])))
    for i in range(list_len):
        loc = cfg.locslist[i]
        scale = cfg.scalelist[i]
        new_train[i*x_shape[0]:(i+1)*x_shape[0],...] = np.random.normal( loc=loc,scale=scale,size=x_shape) + train_x
    return new_train 

def showRand(train_x,length):
    index = np.random.randint(0,length)
    l = 6
    x = [i for i in range(1,train_x.shape[1]+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=l,ncols=1)
    for i in range(0,l):
        aix[i].plot(x,train_x[i*length+index,...])
    plt.show()

