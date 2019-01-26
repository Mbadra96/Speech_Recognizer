from SpeechRecognizer.recognizer import Recognizer
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from SpeechRecognizer.sound_extractor import get_features
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
import pickle
import numpy as np
# song = AudioSegment.from_wav("test.wav")
# new = song.low_pass_filter(2000)
# new.export("f.wav",format="wav")
# x_train = pickle.load(pickle_in)
# x_train = np.array(x_train)
# print(x_train.shape)
# print(get_features("test.wav",type='RNN').shape)
test = Recognizer()
test.start("DNN")
# sr,audio =read("test.wav")
# mfcc = librosa.feature.melspectrogram(y=audio, sr=sr)
#sr dh sampling rate of of y
#y=signal audio time series
# plt.imshow(mfcc,cmap='Greys')
# plt.show()
# mfcc2 = librosa.feature.mfcc(y=audio, sr=sr)
# print(mfcc2.shape)#(20,173)
# print(mfcc.shape)#(128,173)
# print(len(audio))#88064
# pickle_in = open("X.pickle", "rb")
# x_train = pickle.load(pickle_in)
# x_train = np.array(x_train)#110
# print(x_train.size)#191400
# print(np.shape(x_train))#(110,20,87)
# x_train = x_train.reshape(x_train.shape[0], 130, 20)
# print(np.shape(x_train))#
        #x.shape beydelek length of the zeor dimension
        #130
        #20
        #(sample rate/chunck)*seconds
# self.x_train = tf.keras.utils.normalize(self.x_train)