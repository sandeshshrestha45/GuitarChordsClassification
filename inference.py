import glob
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import sounddevice
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

duration = 1   # seconds
sample_rate=22050

loadmodel=load_model('saved_model/chord_classification.hdf5')

labels=['A','A#','A#m','Am','B','BackgroundNoise','Bm','C','C#','C#m','Cm','D','D#','D#m','Dm',
        'E','Em','F','F#','F#m','Fm','G','G#','G#m','Gm']

labelencoder=LabelEncoder()
labels=to_categorical(labelencoder.fit_transform(labels))


def extract_features():
    X = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    #print("X : ", X)
    #print(X.shape)
    sounddevice.wait()
    X = np.squeeze(X)
    #print("squeezedX : ", X)
    #print(X.shape)
    mfccs_features = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    #print(len(mfccs_features))
    #print("mfccs:", mfccs_features.T)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    #print(mfccs_scaled_features)
    return mfccs_scaled_features



if __name__ == "__main__":
    while True:
        features = extract_features()
        prediction=loadmodel.predict(features)
        predicted_label = np.argmax(prediction, axis=1)
        prediction_class = labelencoder.inverse_transform(predicted_label) 
        print ("prediction:", prediction_class)


