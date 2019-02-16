# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:54:58 2019

@author: amanranjan
"""

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features

#path to training data
source   = "pygender\\train_data\\male\\"
#path to save trained model   
dest     = "pygender\\"         
files    = [os.path.join(source,f) for f in os.listdir(source) if 
             f.endswith('.wav')]
 
features = []
for f in files:
    sr,audio = read(f)
    mfcc_f = get_MFCC(sr,audio)
    [features.append(list(i)) for i in mfcc_f]
features = np.asarray(features);

gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag',
        n_init = 3)
gmm.fit(features)
picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"

# model saved as .gmm
cPickle.dump(gmm, open(dest + picklefile,'wb'))
print('modeling completed for gender:',picklefile)