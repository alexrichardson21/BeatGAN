import numpy as np
import random
import re
import sys
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob
from pydub import AudioSegment, silence
from pydub.playback import play
from aubio import source, onset, tempo, pitch, notes
import bpm_extractor as bpm
import ffmpy
import os
from os import path
import requests
import json
import urllib
import soundfile as sf
import io

# import echonest.remix.audio as audio


class AudioGod():
    def __init__(self, shape, bpm):
        self.shape = shape
        self.bpm = bpm
        self.samples_per_beat = 44100 * 60 // self.bpm
        self.sonic_auth_index = 0
        self.auth_codes = json.load('auth.json')
        
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

    def bin_tempos(self, dataset, transformations):
        for filename in glob.glob("datasets/%s/*.mp3" % dataset):
            song = AudioSegment.from_mp3(filename)
            bpm, beats_per_bar, beats = self.sonic_api_analysis(filename)

            for n in range(transformations):

                semitones = random.randint(-4,4) + (random.random() - .5) / 4
                if semitones < 0:
                    tempo_bins = [int(round(bpm, -1)) +
                                n for n in range(-20, 1, 10)]
                elif semitones > 0:
                    tempo_bins = [int(round(bpm, -1)) +
                                n for n in range(0, 21, 10)]
                else:
                    tempo_bins = [int(round(bpm, -1)) +
                                n for n in range(-10, 11, 10)]
                
                new_tempo = np.random.choice(tempo_bins)
                bpm_adjust = new_tempo / bpm + (random.random()-.5) / 250

                tempo_dir = ("./datasets/%s/%dbpm") % (dataset, new_tempo)
                if not path.exists(tempo_dir):
                    os.mkdir(tempo_dir)

                # window = 31 + random.randint(-5, 5) * 2
                # compress = 30 - random.random() * 5

                new_filename = re.sub('[()!@#$]', '-', filename)
                new_filename = new_filename.replace(
                    dataset, dataset + "/%dbpm" % new_tempo).replace(
                    ' ', '-').replace(
                    '.mp3', '_%d.wav' % n)
                
                print("pitch change: %.6f" % semitones)
                print("tempo change: %.6f" % bpm_adjust)

                self.sonic_api_bpm_pitch_adjust(filename, new_filename, bpm_adjust, semitones)

                # FFMPEG
                ##############
                # commands = [
                #     "-acodec", "libmp3lame",
                # #     # "-acodec", "librubberband",
                # #     # "-qscale:a", "2",
                # #     "-ab", "128k",
                #     #     "-ac", "1",' % ,silenceremove=1:0:-50dB, asetrate=%d,atempo=%.6f,aresample=44100,
                #     "-filter:a", "dynaudnorm=g=%d:s=%.5f" % (window, compress),
                # ]

                # ff = ffmpy.FFmpeg(
                #     inputs={filename: None},
                #     outputs={new_filename: commands}
                # )
                # ff.run()

    def sonic_api_analysis(self, filename):
        params = {
            'access_id': self.auth_codes[str(self.sonic_auth_index)],
            'format': 'json'
        }
        files = {
            'input_file': ('song.mp3', open(filename, 'rb'), "multipart/form-data")
        }
        response = requests.post(
            'https://api.sonicAPI.com/analyze/tempo', files=files, params=params)

        if response.status_code == 400:
            self.sonic_auth_index += 1
            response = requests.post(
                'https://api.sonicAPI.com/analyze/tempo', files=files, params=params)

        if response.status_code == 200:
            content = json.loads(response.content)['auftakt_result']
            beats = content['click_marks']
            beats_per_bar = content['clicks_per_bar']
            bpm = round(content['overall_tempo'])
        
            return bpm, beats_per_bar, beats
        
        raise Exception("Bad Sonic API request")

    def sonic_api_bpm_pitch_adjust(self, filename, new_filename, bpm_adjust, semitones):
        params = {
            'access_id': self.auth_codes[str(self.sonic_auth_index)],
            'pitch_semitones': semitones,
            'tempo_factor': bpm_adjust,
            'format': 'wav',
        }
        files = {
            'input_file': ('song.mp3', open(filename, 'rb'), "multipart/form-data")
        }
        response = requests.post(
            'https://api.sonicAPI.com/process/elastique', files=files, params=params)
        
        if response.status_code == 400:
            self.sonic_auth_index += 1
            response = requests.post(
                'https://api.sonicAPI.com/process/elastique', files=files, params=params)
        
        if response.status_code == 200:
            with open(new_filename, mode='bx') as f:
                f.write(response.content)

            print("Saved as: %s" % new_filename)

        raise("Bad Sonic API request")
        # AudioSegment.from_raw(io.BytesIO(response.content)).export(new_filename, format='mp3', bitrate='128k')
        # data, samplerate = sf.read(io.BytesIO(response.content))

        # return data

    def slice_songs(self, dataset, tempo_bin, bars=1):
        
        for filename in glob.glob("datasets/%s/%sbpm/*.wav" % (dataset, tempo_bin)):
            print(filename)

            song = AudioSegment.from_wav(filename)


            bpm, beats_per_bar, beats = self.sonic_api_analysis(filename)
            
            if (round(bpm, -1) != tempo_bin):
                print("Wrong BPM :(")
                continue
            
            print("Actual BPM: %.3f" % bpm)

            downbeats = []
            for beat in beats:
                if beat['downbeat']:
                    downbeats.append(beat)
            
            
            
            print("Slicing %s in %d bar segments" %
                    (os.path.basename(filename), bars))
            
            for beat in range(0, len(beats), beats_per_bar * 2):
                
                # if curr beat tempo is close to desired tempo
                if round(beats[beat]['bpm'], -1) == tempo_bin:
                    
                    # slice by size bars
                    
                    # if slice start not on beat -> move to beat
                    if round(slice_start, -1) != round(beats[beat]['time'] * 44100):
                        slice_start = round(beats[beat]['time'] * 44100)
                    
                    slice_stop = slice_start + samples_per_beat * bars
                    
                    try:
                        bar_slice = song.get_sample_slice(slice_start, slice_stop)
                        
                        slice_start = slice_stop
                        
                        # export slice
                        slice_export_path = "datasets/%s/%dbpm/slices/%s" % (
                            dataset, tempo_bin, os.path.basename(filename).replace(".wav", "_slice%d.mp3" % beat))
                        print(slice_export_path)
                        bar_slice.export(slice_export_path, format="mp3")
                    except:
                        nothing = 0

    def start_songs_at_first_beat(self, dataset):
        for filename in glob.glob("datasets/%s/*.mp3" % dataset):
            
            song = AudioSegment.from_mp3(filename)
            bpm, beats_per_bar, beats = self.sonic_api_analysis(filename)

            samples_per_beat = int(60 / bpm * 44100 * 4)
            start_beat = None
            response = "replay"

            # User input needed for determining first beat of song
            for beat in range(beats_per_bar*2):
                slice_start = round(beats[beat]['time'] * 44100)
                slice_stop = slice_start + samples_per_beat
                bar_slice = song.get_sample_slice(slice_start, slice_stop)
                input('Playing slice %d. Press any key to continue\n' %
                        (beat + 1))

                while response.lower().strip() == 'replay':
                    play(bar_slice + bar_slice)
                    response = input("Is this loop correct? y/n/replay/skip\n")
                if response.lower().strip() == 'skip':
                    break
                elif response.lower().strip() in ['y', 'yes']:
                    start_beat = beat
                    break

            # If first beat -> run n random transformations
            if start_beat:
                song.get_sample_slice(
                    start_sample=start_beat).export(out_f=filename)


if __name__ == "__main__":
    ag = AudioGod(
        shape=(1, 210, 210, 2), 
        bpm=120
    )
    ag.slice_songs('sound_cloud', 110)
