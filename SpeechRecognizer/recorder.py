import pyaudio
import wave
import os
import re
import time
from SpeechRecognizer.audio_player import AudioFile
from SpeechRecognizer.Filter import Filteration
from termcolor import colored
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
cwd = os.path.abspath(os.path.join("", os.pardir))
Dir = cwd+"\\audio"
catdir = cwd+"\\Categories.txt"
f = open(catdir, "r")
CATEGORIES = f.read().replace('\n', ',').split(',')
f.close()


class SoundRecorder:
    def __init__(self, categories=CATEGORIES, directory=Dir, recording_period=RECORD_SECONDS):
        self.categories = categories
        self.dir = directory
        l = []
        for file in os.listdir(self.dir):
            l.append(str(file))
        for cat in self.categories:
            if cat not in l:
                print("the word '", cat, "' does't have a folder but it will be added  :)")
                os.makedirs(self.dir+"\\"+cat)
        self.recording_period = recording_period
        self.ding = AudioFile(self.dir + '\ding.wav')
        self.dong = AudioFile(self.dir + '\dong.wav')

    def get_last_number(self, cat):
        path = os.path.join(self.dir, cat)
        m = 0
        for file in os.listdir(path):
            tmp = int(str(file.replace('.wav', '')))
            if m < tmp:
                m = tmp
        return m

    def record(self, filename, cat):

        print("* you should say the following sentance:", colored(cat,'red'), "after the beep in ")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("GO")

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        self.ding.play()
        for i in range(0, int(RATE / CHUNK * self.recording_period)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("done recording ", cat)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.dong.play()
        print("this is your voice")
        AudioFile(filename).play()
        choice = input("Is your voice okay in the previous recording?(answer with y or n)")
        if choice == "n":
            self.delete_last(cat)
            self.record(filename, cat)

    def record_file(self, filename):
        print("* recording after")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("GO")

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        self.ding.play()
        for i in range(0, int(RATE / CHUNK * self.recording_period)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("done recording ")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.dong.play()
        print("this is your voice")
        #Filteration(filename).highpassfilter(1000)
        #Filteration(filename).lowpassfilter(380)
        #sound = AudioSegment.from_file(filename, format="wav")
        #play(sound)
        AudioFile(filename).play()
        choice = input("Is your voice okay in the previous recording?(answer with y or n)")
        if choice == "n":
           self.record_file(filename)

    def record_file_now(self, filename):

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        self.ding.play()
        for i in range(0, int(RATE / CHUNK * self.recording_period)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.dong.play()

    def delete_last(self, cat):
        last_number = self.get_last_number(cat)
        fi = self.dir + "\\" + cat + "\\" + str(last_number) + ".wav"
        os.remove(fi)

    def record_for_one_category(self, category):
        if category not in self.categories:
            print(category, "is not on my list so please check categories.txt file")
            return
        print("You will have 3 seconds to record")
        time.sleep(2)

        after_last_number = self.get_last_number(category) + 1
        file = self.dir + "\\" + category + "\\" + str(after_last_number) + ".wav"
        self.record(file, category)
        print("Thank You For Helping :)")

    def record_for_all_categories(self):

        print("You will have 3 seconds to record every word")
        print("* you will see a sentance in red and you need to repeat it after the beep, you will listen to it again so please confirm with y or n to make sure it is recorded properly :")
        time.sleep(2)

        for category in self.categories:
            after_last_number = self.get_last_number(category)+1
            file = self.dir + "\\" + category + "\\" + str(after_last_number)+".wav"
            self.record(file, category)
        print("Thank You For Helping :)")

    def delete_last_for_all_categories(self):
        for category in self.categories:
            self.delete_last(category)

        print("Done Deleting")


