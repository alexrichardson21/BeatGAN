import numpy as np
import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob
from pydub import AudioSegment
from aubio import source, onset, tempo, pitch, notes
import bpm_extractor as bpm
import ffmpy


class AudioGod():
    def __init__(self, shape, tempo):
        self.shape = shape
        self.tempo = tempo
        self.samples_per_beat = 44100 * 60 / self.tempo
        
    def load_songs(self, dataset):
        (beats, slices, samples, channels) = self.shape

        num_songs = len(glob.glob('./datasets/%s/*%dbpm.wav' % (dataset, self.bpm)))

        print('loading %d songs ...' % num_songs)
        
        songs = np.zeros((num_songs,) + self.shape)
        len_songs = 0

        for filepath in glob.iglob('./datasets/%s/*%dbpm.wav' % (dataset, self.bpm)):
            try:
                song = AudioSegment.from_wav(filepath)
                silence = song.detect_leading_silence()

                song = song.get_array_of_samples()[silence:]
                
                section_size = self.samples_per_beat * beats

                for i in range(0, len(song)-section_size, section_size):
                    section = song[i:i+section_size]
                    np.reshape(section, shape)
           
                    songs[len_songs] = section
                    len_songs += 1
                    print(".")
            except:
                print("e")
            
        # -1 to 1
        songs = songs / max([songs.max, songs.min])

        return songs[:len_songs]
    
    def mp3_to_wav(self, dataset):
        # mp3 -> wav
        for filename in glob.iglob("datasets/%s/*.mp3" % dataset):
            song = AudioSegment.from_mp3(filename)
            song.export(filename.replace(".mp3", ".wav"), format="wav")

    def change_tempos(self, dataset):
        # make all wavs 120 bpm
        for filename in glob.glob("datasets/%s/*.wav" % dataset):
            tempo_adjust = tempo / bpm.get_file_bpm(filename)
            
            new_filename = filename.replace(dataset, dataset + "_%dbpm" % tempo)
            ff = ffmpy.FFmpeg(
                inputs={filename: None}, 
                outputs={new_filename: ["-filter:a", "atempo=%.10f" % tempo_adjust]}
            )
            ff.run()

    
if __name__ == "__main__":
    ag = AudioGod()
    ag.pydubbin()
    # for song in ag.load_songs((1411200, 2), 'starter'):
    #     bars = len(song) // 44100
    #     for bar in song.reshape((bars, 44100, 2)):
    #         spectrums = ag.bar_as_spectrums(210, 500000, (44100, 2), bar)
    #         plt.plot(spectrums[np.random.randint(len(spectrums))])
    #         plt.show()
    

