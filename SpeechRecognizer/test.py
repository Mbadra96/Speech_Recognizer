from SpeechRecognizer.recognizer import Recognizer
from pydub import AudioSegment
from pydub.effects import low_pass_filter
# song = AudioSegment.from_wav("test.wav")
# new = song.low_pass_filter(2000)
# new.export("f.wav",format="wav")
r = Recognizer(from_file=False,file_name="F:\Work\Shell\Voice Recognition\Virtual_Assistant\\audio\\nova\\6.wav",model="ModelRNN.h5")
r.start(type="RNN")
# import pickle
# import numpy as np
# pickle_in = open("X.pickle", "rb")
# x_train = pickle.load(pickle_in)
# x_train = np.array(x_train)
# print(x_train.shape)
#
# from SpeechRecognizer.sound_extractor import get_features
#
# print(get_features("test.wav",type='RNN').shape)
import librosa
import matplotlib.pyplot as plt
# audio, sr = librosa.load("test.wav")
# mfcc = librosa.feature.melspectrogram(y=audio, sr=sr)
# plt.imshow(mfcc,cmap='Greys')
# plt.show()

#print(mfcc.shape)