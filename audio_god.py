import numpy as np
import random
import re
import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob
from pydub import AudioSegment, silence
from aubio import source, onset, tempo, pitch, notes
import bpm_extractor as bpm
import ffmpy
import os
from os import path
import requests
import json
import urllib

# import echonest.remix.audio as audio


class AudioGod():
    def __init__(self, shape, bpm):
        self.shape = shape
        self.bpm = bpm
        self.samples_per_beat = 44100 * 60 // self.bpm
        
    def load_songs(self, dataset):
        (beats, slices, samples, channels) = self.shape

        num_songs = len(glob.glob('./datasets/%s/%dbpm/*.wav' % (dataset, self.bpm)))

        print('loading %d songs ...' % num_songs)
        
        songs = np.zeros((num_songs,) + self.shape)
        len_songs = 0

        for filepath in glob.iglob('./datasets/%s/%dbpm/*.wav' % (dataset, self.bpm)):
            try:
                # song = AudioSegment.from_wav(filepath)
                # silence = silence.detect_leading_silence(song)

                # song = np.reshape(song.get_array_of_samples(), (-1, 2))#[silence:]
                
                
                
                section_size = self.samples_per_beat * beats

                for slice_start in range(0, len(song)-section_size, section_size):
                    section = song[slice_start:slice_start+section_size]
                    wavfile.write(
                        filepath.replace('.wav', '_%d.wav' % slice_start), 44100, np.array(section))
                    np.reshape(section, self.shape)
           
                    songs[len_songs] = section
                    len_songs += 1
                    print(".")
            except:
                print("e")
            
        # -1 to 1
        songs = songs / max([songs.max, abs(songs.min)])

        return songs[:len_songs]
    
    def mp3_to_wav(self, dataset):
        # mp3 -> wav
        for filename in glob.iglob("datasets/%s/*.mp3" % dataset):
            song = AudioSegment.from_mp3(filename)
            song.export(filename.replace(".mp3", ".wav"), format="wav")

    def change_tempos(self, dataset):
        # make all wavs 120 bpm
        for filename in glob.glob("datasets/%s/*.wav" % dataset):
            bpm_adjust = self.bpm / bpm.get_file_bpm(filename)
            
            new_filename = re.sub('[()!@#$]', '', filename)
            new_filename = new_filename.replace(dataset, dataset + "/%dbpm" % self.bpm).replace(' ','-')
            
            
            ff = ffmpy.FFmpeg(
                inputs={filename: None}, 
                outputs={new_filename: ["-filter:a", "atempo=%.10f" % bpm_adjust]}
            )
            ff.run()

    def bin_tempos(self, dataset, transformations):
        for filename in glob.glob("datasets/%s/*.mp3" % dataset):
            try:
                bpm, beats = self.sonic_api_analysis(filename)
            except:
                print("Sonic API not working")
                continue
            # for beat in beats:
            #     if beat['downbeat']:
            #         slice_start = beat['time']

            for n in range(transformations):
                new_bit_rate = int(44100 * (1 + (random.random()/2 - .5)))
                file_bpm = bpm * (44100 / new_bit_rate)
                tempo_bin = int(round(file_bpm, -1))
                bpm_adjust = tempo_bin / file_bpm

                tempo_dir = ("./datasets/%s/%dbpm") % (dataset, tempo_bin)
                if not path.exists(tempo_dir):
                    os.mkdir(tempo_dir)
                
                window = 31 + random.randint(-5, 5) * 2
                compress = 30 - random.random() * 5

                new_filename = re.sub('[()!@#$]', '-', filename)
                new_filename = new_filename.replace(
                    dataset, dataset + "/%dbpm" % tempo_bin).replace(
                    ' ', '-').replace(
                    '.mp3', '_%d.mp3' % n)
                
                print("pitch change: %d" % new_bit_rate)
                print("tempo change: %.6f" % bpm_adjust)



                # commands = [
                #     # "-acodec", "libmp3lame",
                #     # "-acodec", "librubberband",
                #     # "-qscale:a", "2",
                #     "-ab", "128k",
                #     "-ac", "1",' % ,
                #     "-filter:a", "silenceremove=1:0:-50dB, asetrate=%d,atempo=%.6f,aresample=44100,dynaudnorm=g=%d:s=%.5f" % (
                #         new_bit_rate, bpm_adjust, window, compress),
                # ]

                # ff = ffmpy.FFmpeg(
                #     inputs={filename: None},
                #     outputs={new_filename: commands}
                # )
                # ff.run()

    def sonic_api_analysis(self, filename):
        params = {
            'access_id': 'd0b153ce-692e-4f3c-b439-bddd3fbb1705',
            'format': 'json'
        }
        files = {
            'input_file': ('song.mp3', open(filename, 'rb'), "multipart/form-data")
        }
        response = requests.post(
            'https://api.sonicAPI.com/analyze/tempo', files=files, params=params)

        if response.status_code != 200:
            return None
        
        content = json.loads(response.content)['auftakt_result']
        beats = content['click_marks']
        bpm = round(content['overall_tempo'])
        
        return bpm, beats

    def sonic_api_bpm_pitch_adjust(self, filename, bpm_adjust, semitones):
        params = {
            'access_id': 'd0b153ce-692e-4f3c-b439-bddd3fbb1705',
            'pitch_semitones': semitones,
            'tempo_factor': bpm_adjust,
            'blocking': 'false',
            'format': 'wav',
        }
        files = {
            'input_file': ('song.mp3', open(filename, 'rb'), "multipart/form-data")
        }
        response = requests.post(
            'https://api.sonicAPI.com/process/elastique', files=files, params=params)

        if response.status_code != 200:
            return None
        
        song = urllib.urlretrieve(
            response.content['href'], filename)

        return np.asarray(response.content)

    def slice_songs(self, dataset, tempo_bin, bars=1, save_rate=5):
        for filename in glob.glob("datasets/%s/%sbpm/*.mp3" % (dataset, tempo_bin)):
            song = AudioSegment.from_mp3(filename)
            bpm, beats = self.sonic_api_analysis(filename)
            
            print("Actual BPM: %d" % int(bpm))
            
            downbeats = []
            for beat in beats:
                if beat['downbeat']:
                    downbeats.append(beat)
            # 60000 ms in minute
            samples_per_beat = int(60 / tempo_bin * 44100)
            
            slices = []
        
            for i in range(len(downbeats) // bars):
                slice_start = round(downbeats[i]['time'] * 44100)
                slice_stop = slice_start + samples_per_beat
                slices.append(song.get_sample_slice(slice_start, slice_stop))
                if (i+1) % save_rate == 0:
                    base = os.path.basename(filename)
                    slices[i].export("samples/slices/%s" % base.replace(".mp3", "_slice%d.mp3" % int(i+1)), format="mp3")
            
if __name__ == "__main__":
    ag = AudioGod(
        shape=(1, 210, 210, 2), 
        bpm=120
    )
    # ag.bin_tempos('sound_cloud', 5)
    song = ag.sonic_api_bpm_pitch_adjust('datasets/sound_cloud/6AM.mp3', .75, -6)
    ag.slice_songs('sound_cloud', 80)
    # print(songs)
    

