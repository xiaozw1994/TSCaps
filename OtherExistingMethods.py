import tensorflow as tf
import numpy as np

import keras
import numpy as np
import time
import os
import config as cfg
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn import metrics
import keras_contrib
from utils import reduce_sum
from utils import softmax
from utils import get_shape
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



eposilion = 1e-9

def batch_normal(value,is_training=False,name='batch_norm'):
    if is_training is True:
         return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = True)
    else:
        #
        return tf.contrib.layers.batch_norm(inputs=value,decay=0.9,updates_collections=None,is_training = False)
def Active(x,mode='relu'):
    if mode == 'relu' :
        return tf.nn.relu(x)
    elif mode == 'leaky_relu' :
        return tf.nn.leaky_relu(x,alpha=0.1)
    else:
        return tf.nn.tanh(x)
mode = 'leaky_relu'



def BuildCNNs(X,num_label,is_train=True):
    x1 = tf.contrib.layers.conv1d(X,6,7,padding="SAME",activation_fn=tf.nn.sigmoid)
    x2 = tf.layers.AveragePooling1D(pool_size=3,strides=3)(x1)
    x3 = tf.contrib.layers.conv1d(x2,12,7,padding="SAME",activation_fn=tf.nn.sigmoid)
    x4 = tf.layers.AveragePooling1D(pool_size=3,strides=3)(x3)
    x5 = tf.layers.flatten(x4)
    y = tf.contrib.layers.fully_connected(x5,num_label)
    return y


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

slim = tf.contrib.slim

def Self_Attention1D(x,chanel):
    query = slim.conv1d(x,chanel//2,1,1)
    key = slim.conv1d(x,chanel//2,1,1)
    value = slim.conv1d(x,chanel,1,1)
    b , w , c = get_shape(x)
    key = tf.reshape(key,[-1,chanel//2,w])
    attend = tf.matmul(query,key)
    attend = tf.nn.softmax(attend,axis=-1)
    out = tf.matmul(attend,value)
    #print(out)
    scale = tf.constant(1.0,tf.float32)
    out = scale * out +x
    return out

def DecoderModel(X,num_label,is_train=True):
    ##### first-1
    x1 = tf.contrib.layers.conv1d(X,128,5,stride=1,padding='SAME',activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    rel = prelu(bn1)
    rel = tf.nn.dropout(rel,0.2)
    max1 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    x2 = tf.contrib.layers.conv1d(max1,256,11,stride=1,padding='SAME',activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    rel = prelu(bn2)
    rel = tf.nn.dropout(rel,0.2)
    max2 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    x3 = tf.contrib.layers.conv1d(max2,512,21,stride=1,padding='SAME',activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    rel = prelu(bn3)
    rel = tf.nn.dropout(rel,0.2)
    max3 = tf.layers.max_pooling1d(rel,2,2,padding='SAME') 
    max3 = batch_normal(max3,is_train)
    ##   Attention Mechanism
    ###
    attent = Self_Attention1D(max3,512)
    ful_feature = tf.layers.flatten(attent)
    soft = tf.contrib.layers.fully_connected(ful_feature,num_label,activation_fn=tf.nn.softmax)
    return soft

def BuildFCN(X,num_label,length,is_train=True):
    x1 = tf.contrib.layers.conv1d(X,128,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    ac1 = Active(bn1,mode)
    x2 = tf.contrib.layers.conv1d(ac1,256,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    ac2 = Active(bn2,mode)
    x3 = tf.contrib.layers.conv1d(ac2,128,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    ac3 = Active(bn3,mode)
    avg = tf.layers.average_pooling1d(ac3,length,1)
    fc = tf.layers.flatten(avg)
    fc = tf.contrib.layers.fully_connected(fc,num_label,activation_fn=tf.nn.softmax)
    return fc

def BuildModelMCC(X,num_label,is_train=True):
    x = tf.contrib.layers.conv1d(X,8,5,1,padding="SAME",activation_fn=tf.nn.relu)
    x = tf.layers.max_pooling1d(x,2,2,padding="SAME")
    x = tf.contrib.layers.conv1d(x,8,5,stride=1,padding="SAME",activation_fn=tf.nn.relu)
    x = tf.layers.max_pooling1d(x,2,2,padding="SAME")
    x = tf.layers.flatten(x)
    fully = slim.fully_connected(x,732,activation_fn=tf.nn.relu)
    out = slim.fully_connected(fully,num_label,activation_fn=tf.nn.softmax)
    return out

def BuildMLP(X,num_label,is_train=True):
    x = slim.flatten(X)
    x = slim.dropout(x,0.1)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.2)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.2)
    x = slim.fully_connected(x,500,activation_fn=tf.nn.relu)
    x = slim.dropout(x,0.3)
    x = slim.fully_connected(x,num_label,activation_fn=tf.nn.softmax)
    return x

def baseResidual(X,num_features,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    res1 = slim.conv1d(X,num_features,1,stride=1,padding="SAME",activation_fn=None)
    res1 = batch_normal(res1,is_train)
    block1 = Active(bn3+res1,mode)
    return block1

def BaseBlock3(X,num_features,is_train):
    mode = 'relu'
    x1 = slim.conv1d(X,num_features,8,stride=1,padding="SAME",activation_fn=None)
    bn1 = batch_normal(x1,is_train)
    re1 = Active(bn1,mode)
    x2 = slim.conv1d(re1,num_features,5,stride=1,padding="SAME",activation_fn=None)
    bn2 = batch_normal(x2,is_train)
    re2 = Active(bn2,mode)
    x3 = slim.conv1d(re2,num_features,3,stride=1,padding="SAME",activation_fn=None)
    bn3 = batch_normal(x3,is_train)
    res1 = batch_normal(X,is_train)
    block1 = Active(bn3+res1,mode)
    return block1

###
###
def BuildResNet(X,num_label,length,is_train=True):
    #####BLock 1
    mode = 'relu'
    num_features = 64
    
    ########BLock2
    block1 = baseResidual(X,num_features,is_train)
    block2 = baseResidual(block1,num_features*2,is_train)
    block3 = BaseBlock3(block2,num_features*2,is_train)
    ful = tf.layers.average_pooling1d(block3,length,1)
    ful = slim.flatten(ful)
    ful = slim.fully_connected(ful,num_label,activation_fn=tf.nn.softmax)
    return ful

###############
######## Inception
########

def BottleIncep(X,num_feature,is_train):
    x = slim.conv1d(X,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.tanh)
    kernles = [3, 5, 8, 11, 17]
    values = []
    for i in range(len(kernles)):
        values.append(slim.conv1d(x,num_feature,kernles[i],stride=1,padding="SAME",activation_fn=tf.nn.tanh))
    conv1 = tf.layers.max_pooling1d(x,2,1,padding="SAME")
    conv2 = slim.conv1d(conv1,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.tanh)
    values.append(conv2)
    res = tf.concat(values,axis=-1)
    res = batch_normal(res,is_train)
    return Active(res)

def baseIncep(X,num_feature,is_train):
    x = X
    kernles = [3, 5, 8, 11, 17]
    values = []
    for i in range(len(kernles)):
        values.append(slim.conv1d(x,num_feature,kernles[i],stride=1,padding="SAME",activation_fn=tf.nn.tanh))
    conv1 = tf.layers.max_pooling1d(x,2,1,padding="SAME")
    conv2 = slim.conv1d(conv1,num_feature,1,stride=1,padding="SAME",activation_fn=tf.nn.tanh)
    values.append(conv2)
    res = tf.concat(values,axis=-1)
    res = batch_normal(res,is_train)
    return Active(res)

def Inception(X,num_label,length,is_train=True):
    bottle = BottleIncep(X,32,is_train)
    depth = 4
    x = bottle
    for i in range(depth):
        x = baseIncep(x,32,is_train)
    res = slim.conv1d(X,32*6,1,1,padding="SAME",activation_fn=None)
    res = batch_normal(res,is_train)
    x = res+x
    x = Active(x)
    x = tf.layers.average_pooling1d(x,length,1)
    ful = slim.flatten(x)
    ful = slim.fully_connected(ful,num_label,activation_fn=tf.nn.softmax)
    return ful


def Totalcount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
        # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print("The Total params:",total_parameters/1e6)


def TestGender():
    class_name = 'Crop'
    data_dir = os.path.join(cfg.data_raw_main, class_name)
    x_train , y_train = cfg.readucr(data_dir+"/"+class_name+"_TRAIN")
    x_test, y_test = cfg.readucr(data_dir+"/"+class_name+"_TEST")
    num_classes = len(np.unique(y_test))
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(num_classes-1)
    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    length = 46
    batch_size =  50
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_test.mean(axis=0))/(x_test.std(axis=0))
    x_train = x_train.reshape((-1,length,1))
    x_test = x_test.reshape((-1,length,1))
    total_train_epoch = x_train.shape[0] // batch_size
    total_test_epoch = x_test.shape[0] // batch_size
    X = tf.placeholder(tf.float32,[None,length,1])

    y = DecoderModel(X, num_classes)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    Totalcount()
    value = 0
    start_time = time.time()
    pt = 0
    for j in range(total_test_epoch):
        batch_x = x_test[j*batch_size:(j+1)*batch_size,:,:]
        batch_y = Y_test[j*batch_size:(j+1)*batch_size,:]
        pt += sess.run(y,feed_dict={X:batch_x}) 
    spend = time.time() - start_time
    print("Total Spend Time:%.5fs, avg Time:%.5fs"%(spend,spend/total_test_epoch))
    sess.close()
TestGender()
