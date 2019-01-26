from SpeechRecognizer.sound_extractor import SoundExtraction
from SpeechRecognizer.neural_network import RNN,plot
from SpeechRecognizer.recognizer import Recognizer
S = SoundExtraction()
S.create_training_data(type='RNN')
T = RNN()
history = T.train()
plot(history)
test = Recognizer()
test.start("DNN")
