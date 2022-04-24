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
outdir='/scratch/Hammad/work/data_Riemann_final/data/'
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

###########The model##################
inputs = Input(shape=(50, 50, 1))
x = Conv2D(16, (3, 3),padding='same',  strides=1,  activation='relu')(inputs)
x = Conv2D(16, (3, 3),padding='same',  strides=1,  activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.25)(x)
x=Conv2D(32, (3, 3),  activation='relu')(x)
x=Conv2D(32, (3, 3),  activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.25)(x)
x=Conv2D(64, (2, 2),  activation='relu')(x)
x=Conv2D(64, (2, 2),  activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.25)(x)
x= Flatten()(x)
x=Dense(265, activation='relu')(x)
outputs = Dense(2, activation='sigmoid')(x)

model_cnn =  Model(inputs,outputs)

########### Handle the labels ##################

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

##########Compile the model ###########
model_cnn.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-03), metrics=['accuracy'])
history_cnn = model_cnn.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=256,shuffle=True, verbose=1)
############################
pd.DataFrame(history_cnn.history).plot(xlabel='epochs',title='Sig vs oct (normal images)');

####Evaluate the model #############
scores = model_cnn.evaluate(x_test, y_test, verbose=0)

print("%s: %.2f%%" % (model_cnn.metrics_names[1], scores[1]*100))

##########################
score=model_cnn.predict(x_test);
fpr, tpr, thresholds =roc_curve(y_test.ravel(),score.ravel());
######Plot ROC #############
plt.style.use('default')
plt.plot([0,1],[0,1],'k-',linewidth=1);
plt.plot(fpr,tpr,linewidth=2,label='Normal images (Negative Octet)= {:.2f}%'.format(float(auc(fpr, tpr))*100));
plt.xlabel(r'False Singlet',fontsize=15,c='k');
plt.ylabel(r'True Singlet',fontsize=15,c='k');
plt.grid(linestyle='--',c='k')
plt.legend(loc='best');
plt.tick_params(axis='both',labelsize=13)
plt.tight_layout()
plt.show()

######5 folds cross validation ########

def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train, y_train, validation_split=0.2, epochs=20, batch_size=256,shuffle=True, verbose=1)
    ss=model.predict(x_test);
    fpr, tpr, _ =roc_curve(y_test.ravel(),ss.ravel(),pos_label={0,1});
    return  model.evaluate(x_test,y_test)[-1],fpr,tpr

folds = KFold(n_splits=5)
score,fpr,tpr = [],[],[]
for train_idx,test_idx in folds.split(x_data):
    X_train,X_test, y_train,y_test = x_data[train_idx],x_data[test_idx],y_data[train_idx],y_data[test_idx]  
    a,b,c = get_score(model_cnn,X_train,X_test, y_train,y_test)
    score.append(a)
    fpr.append(b)
    tpr.append(c)
print(score_1)


 
