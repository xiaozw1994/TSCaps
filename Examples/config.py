
import numpy as np
import os 

from keras.models import Model
from keras.utils import np_utils
import pandas as pd
import keras 
from keras.callbacks import ReduceLROnPlateau
      
def readucr(filename):
    data = np.loadtxt(filename+".tsv", delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

data_raw_main = 'dataset/UCRArchive_2018/'
class_names = ['DodgerLoopWeekend', 'CricketX', 'FiftyWords', 
'FaceFour', 'GestureMidAirD1', 'Wine', 'MixedShapesSmallTrain', 'InsectEPGSmallTrain',
'Rock', 'MiddlePhalanxOutlineAgeGroup', 'StarLightCurves', 'ChlorineConcentration', 
'CBF', 'InsectEPGRegularTrain', 'TwoLeadECG', 'ECGFiveDays', 'Chinatown', 'DodgerLoopGame', 
'ToeSegmentation2', 'ElectricDevices', 'Trace', 'Haptics', 'Symbols', 'Lightning2', 
'MixedShapesRegularTrain', 'LargeKitchenAppliances', 'ShapeletSim', 'HouseTwenty', 
'Mallat', 'OliveOil', 'HandOutlines', 'Strawberry', 'MoteStrain', 'GunPoint', 'EOGHorizontalSignal',
 'BirdChicken', 'PigCVP', 'OSULeaf', 'GesturePebbleZ1', 'FreezerSmallTrain', 'SmallKitchenAppliances',
  'ECG5000', 'Fungi', 'UMD', 'GesturePebbleZ2', 'UWaveGestureLibraryZ', 'ShapesAll', 'Plane', 'Lightning7', 
  'DistalPhalanxTW', 'SyntheticControl', 'Fish', 'GunPointOldVersusYoung', 'ECG200', 'InsectWingbeatSound', 
  'DistalPhalanxOutlineCorrect', 'GestureMidAirD3', 'Beef', 'ToeSegmentation1', 'PigArtPressure', 'Phoneme',
   'RefrigerationDevices', 'SmoothSubspace', 'FordB', 'ArrowHead', 'MedicalImages', 'SemgHandGenderCh2', 
   'Adiac', 'SemgHandSubjectCh2', 'PowerCons', 'UWaveGestureLibraryAll', 'DiatomSizeReduction', 'WordSynonyms', 
   'GestureMidAirD2', 'SemgHandMovementCh2', 'Herring', 'SonyAIBORobotSurface1', 'PickupGestureWiimoteZ',
    'EthanolLevel', 'PhalangesOutlinesCorrect', 'SonyAIBORobotSurface2', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'Ham', 
    'PigAirwayPressure', 'PLAID', 'GunPointAgeSpan', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 
    'TwoPatterns', 'AllGestureWiimoteZ', 'Coffee', 'NonInvasiveFetalECGThorax2', 'DodgerLoopDay', 'InlineSkate', 
    'Earthquakes', 'Car', 'Crop', 'DistalPhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'SwedishLeaf',
     'NonInvasiveFetalECGThorax1', 'Wafer', 'FaceAll', 'MelbournePedestrian', 'Computers', 'ShakeGestureWiimoteZ', 
     'WormsTwoClass', 'EOGVerticalSignal', 'GunPointMaleVersusFemale', 'Worms', 'FreezerRegularTrain', 'ScreenType', 'CinCECGTorso', 
     'FordA', 'UWaveGestureLibraryX', 'BME', 'ProximalPhalanxOutlineAgeGroup', 'BeetleFly',
      'ACSF1', 'Meat', 'FacesUCR', 'CricketZ', 'Yoga', 'MiddlePhalanxTW', 'CricketY', 'ItalyPowerDemand', 
      'UWaveGestureLibraryY']
single_class = 'Coffee'
length = 286 
sota = 0.8500
class Config90(object):
    def __init__(self):
        self.batch_size =  14
        self.num_label = 2
        self.stddev = 0.01
        self.decay = 0.0005 * length 
        self.new_decay = 0.0005 *length           
        ###
        self.init_seq = 0.9
        self.init_sub = 1 - self.init_seq
        ##
        self.lamdaset = 0.5

