import pyaudio
import wave


class AudioFile:
    chunk = 1024

    def __init__(self, file):

        self.file = file

    def play(self):
        wf = wave.open(self.file, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True)
        """ Play entire file """
        data = wf.readframes(self.chunk)
        while data:
            stream.write(data)
            data = wf.readframes(self.chunk)
        """ Graceful shutdown """
        stream.stop_stream()
        stream.close()
        p.terminate()

