import mne
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import load
import tensorflow
from tensorflow.keras import backend as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from math import sqrt, log, exp
import statistics
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from data_gen import DataGenerator
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
#from tensorflow.keras import backend as K

# Parameters
params = {'dim': (26,1280),
          'batch_size': 256,
          'n_classes': 2,
          'shuffle': False}

# Datasets

train_arr=[]
test_arr=[]

y_all=np.load('y_all.npy')

labels={}

cnt=0
for i in range(1,139661):
    train_arr.append('sample'+str(i))
    labels['sample'+str(i)]=y_all[cnt]
    cnt=cnt+1

y_all_test=[]

for i in range(139661,174577):
    test_arr.append('sample'+str(i))
    labels['sample'+str(i)]=y_all[cnt]
    y_all_test.append(y_all[cnt])
    cnt=cnt+1
y_all_test=np.array(y_all_test)

partition = {'train': train_arr, 'validation': test_arr}
#labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dense(25, activation='relu'))
model.add(LSTM(128, return_sequences=True))
model.add(Dense(25, activation='relu'))
model.add(LSTM(128))
model.add(Dense(25, activation='relu'))
model.add(Dense(2, activation='softmax'))
opti = Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opti, metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=4,epochs=45,verbose=2)
#y_train=model.predict_generator(training_generator)
#print(len(y_train))
y_pred=model.predict_generator(validation_generator)
#for i in y_pred[-256:]:
#    print(i)
y_pred = np.argmax(y_pred, axis=1)
#print(len(y_pred))
#print(confusion_matrix(y_all_test,y_pred[:len(y_all_test)]))
unique, counts = np.unique(y_all_test[:len(y_pred)], return_counts=True)
print(unique,counts)
print(confusion_matrix(y_all_test[:len(y_pred)],y_pred))
#print(confusion_matrix(validation_generator.classes, y_pred))
#train_acc=model.history.history['accuracy']
#val_acc=model.history.history['val_accuracy']
#print(train_acc)
#print(val_acc)
