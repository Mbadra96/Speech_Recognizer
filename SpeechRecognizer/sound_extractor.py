import librosa
import numpy as np
import os
import pickle
import random
import tensorflow as tf

cwd = os.path.abspath(os.path.join("", os.pardir))
Dir = cwd+"\\audio"

catdir = cwd+"\\Categories.txt"
f = open(catdir, "r")
CATEGORIES = f.read().replace('\n', ',').split(',')
f.close()


class SoundExtraction:
    def __init__(self, categories=CATEGORIES, directory=Dir):
        self.training_data = []
        self.categories = categories
        self.dir = directory

    def create_training_data(self,type='DNN'):
        for category in self.categories:

            path = os.path.join(self.dir, category)
            class_num = self.categories.index(category)

            for file in os.listdir(path):
                file = self.dir+"\\"+category+"\\"+file
                try:
                    feature_vector = get_features(file, type)
                    self.training_data.append([feature_vector, class_num])
                except Exception as e:
                    pass

        random.shuffle(self.training_data)
        x = []
        y = []

        for features, label in self.training_data:
            x.append(features)
            y.append(label)

        pickle_out = open("X.pickle", "wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open("Y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

        print("Done Extracting data")


def get_features(file, type='DNN'):
    audio, sr = librosa.load(file,mono=True)
    # we extract mfcc feature from data
    if type is 'DNN':
        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    elif type is 'RNN':
        mfcc = librosa.feature.mfcc(y=audio, sr=sr)
        pad_width = 87 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

