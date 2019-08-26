import librosa as lb
import glob
import os
import numpy as np
from sklearn.svm import SVC
path = "data/test/"


def prepare_test():
    X_test = []
    Y_test = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        label=filename.split('\\', 1)[1].split('_')[0]

        y, sr = lb.load(filename)
        arr = lb.feature.mfcc(y=y, sr=sr)
        arr =np.reshape(arr,len(arr)*len(arr[0]))
        zeros = np.zeros(800)
        zeros[0:len(arr)] = arr
        X_test.append(zeros)
        Y_test.append(int(label))

    X_test=np.array(X_test)
    Y_test=np.array(Y_test)
    return X_test,Y_test

def SVM_test(svclassifier):
    acc=0
    print("Preparing train array")
    X_test, Y_test = prepare_test()
    print(X_test.shape)
    print(Y_test.shape)
    print("SVM Classifier Training")
    Test=svclassifier.predict(X_test)

    for i in range(0,len(Y_test)):
        if(Test[i]==Y_test[i]):
            acc=acc+1
    acc=acc/len(Y_test)*100
    return acc


