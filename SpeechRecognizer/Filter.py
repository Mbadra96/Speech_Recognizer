import numpy as np
import scipy as sp
import scipy.fftpack
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
import matplotlib.pyplot as plt
class Filteration:
    def __init__(self, file):
        self.file = file
        self.frequency,self.array = read(self.file)
        self.signal_length=len(self.array)
        if self.signal_length ==2:
            self.array=self.array.sum(axis=1) /2
        self.N=self.array.shape[0]
        self.Ts=1.0/self.frequency
        self.secs= self.N/ float(self.frequency)
        self.time_index=sp.arange(0,self.secs,self.Ts)
        self.FFT=abs(sp.fft(self.array))
        self.FFT_side =self.FFT[range(self.N//2)]
        self.freqs=sp.fftpack.fftfreq(self.array.size,self.time_index[1]-self.time_index[0])
        self.freqs_side=self.freqs[range(self.N//2)]
    def highpassfilter(self,cutoff_frequency):
        b, a = signal.butter(5, cutoff_frequency / (self.frequency / 2), btype='highpass')  # ButterWorth filter 4350
        filteredSignal = signal.lfilter(b, a, self.array).astype(dtype= np.int16)
        write(self.file, self.frequency, filteredSignal)  # Saving it to the file.
    def lowpassfilter(self,cutoff_frequency):
        b, a = signal.butter(5, cutoff_frequency / (self.frequency / 2), btype='lowpass')  # ButterWorth filter 4350
        filteredSignal = signal.lfilter(b, a, self.array).astype(dtype= np.int16)
        write(self.file, self.frequency, filteredSignal)  # Saving it to the file.
    def addnoise(self):
         GuassianNoise = np.random.rand(len(self.FFT))  # Adding guassian Noise to the signal.
         NewSound = GuassianNoise + self.array
         write(self.file, self.frequency, NewSound)
    def plotting_signal(self):
        plt.plot(self.time_index,self.array,"g")
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
    def plotting_fft(self):
        plt.plot(self.freqs,self.FFT,"r")
        plt.show()
    def plotting_fft_half(self):
        plt.plot(self.freqs_side,self.FFT_side,"r")
        plt.show()











