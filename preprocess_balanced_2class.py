import mne
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import save
from sklearn.utils import shuffle
import random

def generate_samples(t1,t2,file): #returns total samples (*256) given 2 times in seconds
    raw = mne.io.read_raw_edf(file, verbose='error')
    t1=round(t1,4)
    t2=round(t2,4)
    #print(t1, t2)
    #if raw.info['sfreq']==256.0:
    #print(raw.info)
    data,times=raw[:,round(t1*raw.info['sfreq'],4):round(t2*raw.info['sfreq'],4)]
    #print(times)
    data=data[:26]
    data=np.array(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    #print(data.shape)
    #print(t1,t2)
    return data

def generate_5_samples(t1,t2,file): #generate samples of 5 seconds specifically
    print(t1,t2,file)
    all_5_samples=[]
    raw = mne.io.read_raw_edf(file, verbose='error')
    times=np.arange(t1,t2-(5.0-1.0))
    for i in times:
        if round(i,4)+5.0<t2:
            all_5_samples.append(generate_samples(round(i,4),round(i,4)+5.0,file))
    #print(len(all_5_samples))
    all_5_samples=np.array(all_5_samples)
    print(all_5_samples.shape)
    return all_5_samples

PATH="edf/train/"
PATH2="normalabnormaleeg/edf/train/normal/"
PATH_DATA="_DOCS/ref_train.txt"
lines=""
with open(PATH_DATA) as file1:
    lines=file1.readlines()
patient_dict={}
for line in lines:
    all_data=(line.rstrip()).split(" ")
    if all_data[0] not in patient_dict:
        patient_dict[all_data[0]]=[]
    if all_data[3]=='bckg':
        label=0
    elif all_data[3]=='seiz':
        label=2
    patient_dict[all_data[0]].append([float(all_data[1]),float(all_data[2]),label])
cnt=0
print(len(lines))

#print(fname)

X_all=[]
y_all=[]

candidate_sp=['00006529_s002_t003', '00006514_s012_t006', '00011999_s007_t012', '00008053_s003_t009', '00006083_s005_t000', '00007128_s002_t005', '00012262_s006_t004', '00011999_s007_t013', '00011999_s007_t016', '00011999_s007_t007', '00013145_s001_t001', '00006529_s002_t002', '00008018_s004_t006', '00006107_s001_t001', '00006413_s006_t002', '00007797_s003_t002', '00012262_s006_t007', '00012262_s005_t003', '00006514_s010_t005', '00013085_s002_t014', '00007128_s002_t003', '00007235_s004_t001', '00002886_s006_t002', '00008018_s004_t008', '00006648_s002_t002', '00007802_s002_t008']

candidate_n=['00009416_s001_t000', '00008641_s001_t000', '00003240_s002_t001', '00006213_s002_t000', '00001958_s004_t000', '00008377_s001_t000', '00008301_s001_t000', '00008818_s002_t003', '00009048_s001_t000', '00007757_s003_t001', '00009573_s001_t000', '00009645_s001_t000', '00008461_s001_t001', '00008401_s001_t000', '00008477_s002_t007', '00008477_s003_t001', '00008092_s003_t001', '00010590_s002_t001', '00010589_s001_t001', '00010175_s001_t001', '00006788_s009_t005', '00009782_s002_t000', '00009709_s003_t008', '00009327_s002_t000', '00009328_s002_t009', '00010277_s001_t000']

print(candidate_sp)
print(candidate_n)

for file2 in Path(PATH).glob('**/*.edf'):
    file3=str(file2).split("/")
    file4=file3[-1].split(".edf")[0]
    if file4 in patient_dict.keys():
        if file4 in candidate_sp:
            print(file4)
            print(patient_dict[file4])
            for patient_record in patient_dict[file4]:
                start=patient_record[0]
                end=patient_record[1]
                label=patient_record[2]
                if end-start>=5.0 and label==0:
                    samples=generate_5_samples(start,end,file2)
                    for s in samples:
                        X_all.append(s)
                        y_all.append(label)
            cnt=cnt+1

for file2 in Path(PATH2).glob('**/*.edf'):
    raw = mne.io.read_raw_edf(file2, verbose='error')
    data,times=raw[:,:]
    file3=str(file2).split("/")
    file4=file3[-1].split(".edf")[0]
    if file4 in candidate_n:
        print(file4)
        start=0
        end=data.shape[1]/256.0
        label=1
        if end-start>=5.0:
            samples=generate_5_samples(start,end,file2)
            for s in samples:
                X_all.append(s)
                y_all.append(label)
        cnt=cnt+1

X_all=np.array(X_all)
y_all=np.array(y_all)

X_all,y_all=shuffle(X_all,y_all,random_state=0)

print(X_all.shape)
print(y_all.shape)
np.save("X_all4.npy",X_all)
np.save("y_all4.npy",y_all)
print(cnt)
