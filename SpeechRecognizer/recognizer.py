from tensorflow.python.keras.models import load_model
import numpy as np
from SpeechRecognizer.recorder import SoundRecorder
from SpeechRecognizer.sound_extractor import get_features
import os
cwd = os.path.abspath(os.path.join("", os.pardir))
catdir = cwd+"\\Categories.txt"
file = open(catdir, "r")
CATEGORIES = file.read().replace('\n', ',').split(',')
file.close()


class Recognizer:
    def __init__(self, from_file=False, file_name="", model="Model.h5", time=2):
        self.model = load_model(model)
        self.file = "test.wav"
        self.categories = CATEGORIES
        if from_file:
            if file_name is "":
                print("You have not put a file path so you will record one now as a test.wav ")
                self.recorder = SoundRecorder(recording_period=time)
            else:
                self.file = file_name
        else:
            self.recorder = SoundRecorder(recording_period=time)

    def test_file(self,type="DNN"):
        test_x = np.array(get_features(self.file,type))
        if type is "DNN":
            test_x = test_x.reshape((1, len(test_x)))
        elif type is "RNN":
            test_x = test_x.reshape((1, test_x.shape[1],test_x.shape[0]))
        predicted = self.model.predict(test_x)
        predicted = predicted[0]
        r,predicted = predicted[np.argmax(predicted)]*100,self.categories.__getitem__(np.argmax(predicted))
        print("The Model predicted: ", predicted,"  with :",r," accuracy")

    def start(self,type="DNN"):
        self.recorder.record_file(self.file)
        test_x = np.array(get_features(self.file,type))
        if type is "DNN":
            test_x = test_x.reshape((1, len(test_x)))
        elif type is "RNN":
            test_x = test_x.reshape((1, test_x.shape[1],test_x.shape[0]))
        predicted = self.model.predict(test_x)
        predicted = predicted[0]
        r, predicted = predicted[np.argmax(predicted)] * 100, self.categories.__getitem__(np.argmax(predicted))
        print("The Model predicted: ", predicted, "  with :", r, " accuracy")

    def start_now(self,type="DNN"):
        self.recorder.record_file_now(self.file)
        test_x = np.array(get_features(self.file,type))
        if type is "DNN":
            test_x = test_x.reshape((1, len(test_x)))
        elif type is "RNN":
            test_x = test_x.reshape((1, test_x.shape[1],test_x.shape[0]))
        predicted = self.model.predict(test_x)
        predicted = predicted[0]
        r, predicted = predicted[np.argmax(predicted)] * 100, self.categories.__getitem__(np.argmax(predicted))
        return r, predicted



