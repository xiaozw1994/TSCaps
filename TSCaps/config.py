import os
import time 
import numpy as np

data_files = '/home/josen/deep learning/Caps Time Series/dataset/UCRArchive_2018'

#
#  Noising Configure
loc = 0.05
scale = 0.05
training_name = 'Beef'
length = 470
batch_size = 30
expand = 1.0
locslist = [0.04,0.05,0.06,0.07,0.08,0.09,0.10] 
scalelist = [0.04,0.05,0.06,0.07,0.08,0.09,0.10] 
locslist = [i*expand for i in locslist]
scalelist = [i*expand for i in scalelist]
class Config(object):
    def __init__(self):
        self.stddev = 0.01
        self.decay = 0.0005 * length 
        self.new_decay = 0.0005 *length           
        ###
        self.init_seq = 0.9
        self.init_sub = 1 - self.init_seq
        ##
        self.lamdaset = 0.5

