import warnings
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm, trange
import glob
import os, sys
import time
from matplotlib.colors import LogNorm
import matplotlib 
import tensorflow as tf
import keras as ks
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import TensorBoard
from numpy.random import permutation
import pandas as pd
#from IPython.display import Math, HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import  label_binarize, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import auc
from tensorflow.keras import Model
import keras.backend as k
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.....')
    sys.exit()
else:
    print('Default GPU device :{}'.format(tf.test.gpu_device_name()))
####################Upload the input images#######################
outdir='data/'
image_S = np.load(outdir+'Image_s.npz',allow_pickle=True)['arr_0'][:100000]
image_O = np.load(outdir+'Image_o.npz',allow_pickle=True)['arr_0'][:100000]
image_bkg = np.load(outdir+'Image_bkg.npz',allow_pickle=True)['arr_0'][:100000]
imageR_S = np.load(outdir+'ImageR1_s.npz',allow_pickle=True)['arr_0'][:100000]
imageR_O = np.load(outdir+'ImageR1_o.npz',allow_pickle=True)['arr_0'][:100000]
imageR_bkg = np.load(outdir+'ImageR1_bkg.npz',allow_pickle=True)['arr_0'][:60000]

###########################################
x_data = np.concatenate((image_S,image_O))
y_data = np.array([1]*len(image_S)+[0]*len(image_O))

#########Split into train and test##################
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,stratify=y_data,shuffle=True, test_size=0.2)

###########Normalize the  inputs ######################
scaler = MinMaxScaler()
X_train_norm,X_test_norm=[],[]

for i in tqdm(range(len(X_train))):
    X = scaler.fit_transform(X_train[i])
    X_train_norm.append(X)
   
for j in tqdm(range(len(X_test))):
    y = scaler.fit_transform(X_test[j])
    X_test_norm.append(y)

#############################################
x_train= np.array(X_train_norm).reshape(np.array(X_train_norm).shape + (1,)).astype('float32')
x_test= np.array(X_test_norm).reshape(np.array(X_test_norm).shape + (1,)).astype('float32')

#######################################


def build_classifier(lr=0.001,n1=30,n2=30,n3=30,n4=30,n5=30,n6=30,k1=(2,2),k2=(2,2),k3=(2,2),k4=(2,2),k5=(2,2),k6=(2,2)):
    classifier = Sequential()
    
    classifier.add(Conv2D(n1, k1, input_shape=(50, 50, 1),padding='same',  strides=1, activation='relu'))
    classifier.add(Conv2D(n2, k2, padding='same',  strides=1,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))

    classifier.add(Conv2D(n3, k3, activation='relu'))
    classifier.add(Conv2D(n4, k4, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))


    classifier.add(Conv2D(n5, k5,  activation='relu'))
    classifier.add(Conv2D(n6, k6, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.25))



    classifier.add(Flatten())
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2, activation='softmax'))
 
    classifier.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['accuracy'])
 
    return classifier
    
    
###############################
classifier = KerasClassifier(build_fn = build_classifier)

parameters=dict()
parameters['lr']= [0.00001,0.00005,0.0001,0.0005,0.001,0.005]
parameters['n1']=np.linspace(10,65,55,dtype=int)
parameters['n2']=np.linspace(10,65,55,dtype=int)
parameters['n3']=np.linspace(10,65,55,dtype=int)
parameters['n4']=np.linspace(10,65,55,dtype=int)
parameters['n5']=np.linspace(10,65,55,dtype=int)
parameters['n6']=np.linspace(10,65,55,dtype=int)
parameters['k1']=[(2,2),(3,3),(4,4),(5,5)]
parameters['k2']=[(2,2),(3,3),(4,4),(5,5)]
parameters['k3']=[(2,2),(3,3),(4,4),(5,5)]
parameters['k4']=[(2,2),(3,3),(4,4),(5,5)]
parameters['k5']=[(2,2),(3,3),(4,4),(5,5)]
parameters['k6']=[(2,2),(3,3),(4,4),(5,5)]

###########Randomize the gridover the given parameters 20 times ##########
grid_search = RandomizedSearchCV(classifier, parameters, cv=20,n_iter=100)
classifier_GS = grid_search.fit(x_train, y_train,validation_split=0.2,verbose=0)

######################## summarize result#################

print('Best Score: %s' % classifier_GS.best_score_)
print('Best Hyperparameters: %s' % classifier_GS.best_params_)
    
