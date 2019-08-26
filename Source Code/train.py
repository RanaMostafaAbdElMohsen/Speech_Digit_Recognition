import librosa as lb
import glob
import os
import numpy as np
from sklearn.svm import SVC
path = "data/train/"


def prepare_train():
    X_train = []
    Y_train = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        label=filename.split('\\', 1)[1].split('_')[0]

        y, sr = lb.load(filename)
        arr = lb.feature.mfcc(y=y, sr=sr)

        arr =np.reshape(arr,len(arr)*len(arr[0]))
        zeros=np.zeros(800)
        zeros [0:len(arr)]=arr
        X_train.append(zeros)
        Y_train.append(int(label))
        # print(arr)
        # print(arr.shape)
        # print(label)
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    return X_train,Y_train

def SVM_train():

    print("Preparing train array")
    X_train, Y_train = prepare_train()
    print(X_train.shape)
    print(Y_train.shape)
    print("SVM Classifier Training")
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    return svclassifier

# svclassifier= SVM_train()


