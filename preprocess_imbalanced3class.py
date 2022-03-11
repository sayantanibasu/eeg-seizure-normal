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
#from generator import generate_5_samples
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from numpy import load

X_all=np.load("X_all2.npy")
y_all=np.load("y_all2.npy")

X_all=X_all[:20000]
y_all=y_all[:20000]

un,counts=np.unique(y_all,return_counts=True)
total=len(y_all)

print(un,counts,total)

#cw={0: 5645/1411, 1:5645/4514}
cw={0:total/counts[0],1:total/counts[1],2:total/counts[2]}
print(cw)

numchannels=26

n_gpus=4
devices=tensorflow.config.experimental.list_physical_devices('GPU')
devices_n=[m.name.split("e:")[1] for m in devices]

s=tensorflow.distribute.MirroredStrategy(devices=devices_n[:n_gpus])

with s.scope():
    kFold = StratifiedKFold(n_splits=5)
    for train, test in kFold.split(X_all, y_all):
        model = None
        model = Sequential()
        model.add(LSTM(128, return_sequences=True))
        model.add(Dense(25, activation='relu'))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dense(25, activation='relu'))
        model.add(LSTM(128))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        opti = Adam(learning_rate=0.001*n_gpus)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opti, metrics=['accuracy'])
        X_all_train=X_all[train]
        X_all_test=X_all[test]
        for q in range(numchannels):
            X_all_train[:,q]=stats.zscore(X_all_train[:,q], axis=None)
            X_all_test[:,q]=stats.zscore(X_all_test[:,q], axis=None)
        model.fit(X_all_train, y_all[train], validation_data=(X_all_test, y_all[test]), epochs=25, batch_size=256*n_gpus, verbose=2, class_weight=cw)
        y_pred=model.predict_classes(X_all_test)
        train_acc=model.history.history['accuracy']
        val_acc=model.history.history['val_accuracy']
        print(train_acc)
        print(val_acc)
        print(classification_report(y_all[test], y_pred))
        print(confusion_matrix(y_all[test], y_pred))
