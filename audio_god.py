import glob
import time
import io
import json
import os
import random
import re
import sys
import urllib
import sox
from os import path

import ffmpy
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
from aubio import notes, onset, pitch, source, tempo
from pydub import AudioSegment, silence
from pydub.playback import play
from scipy.io import wavfile

# Preprocessing Order:
# 1. Soundcloud Playlist Download
# 2. Start Songs at First Beat
# 3. Random Transformations and Bin Tempos
# 4. Slice Songs at Bar Intervals
# 5. Load slices


class AudioGod():
    def __init__(self, dataset, tempo, bars, shape):
        # self.samples_per_beat = 44100 * 60 // self.bpm
        self.transformations = 2
        self.channels = 2
        self.sample_rate = 44100
        self.dataset = dataset
        self.tempo = tempo
        self.bars = bars
        self.shape = shape
        self.samples_per_bar = self.sample_rate * 60 // self.tempo * 4

        self.sonic_auth_index = 3
        with open('auth.json') as f:
            self.auth_codes = json.load(f)
        

    ## Main Functions
    ################
    def preprocess(self):
        print("MAKING TRANSFORMATIONS\n")
        self.bin_tempos()
        print("SLICING SONGS\n")
        self.slice_songs_by_bars()
    
    def bin_tempos(self, out_format='wav'):
        
        for in_format in ['mp3', 'wav']:
            print('Checking %s files ...' % in_format)
            
            for filename in glob.glob("datasets/%s/*.%s" % (self.dataset, in_format)):
                basefilename = os.path.basename(filename)
                print(basefilename)
                
                song = AudioSegment.from_file(filename, format=in_format)
                
                if song.duration_seconds > 60 * 5:
                    print('\tError: Song longer than 5 minutes')
                    continue
                
                print('\tAnalyzing song ...')
                bpm, beats_per_bar, beats = self.tempo_analysis(filename)

                # randomly transform n times
                self.transform(bpm, filename, in_format, out_format)

    def transform(self, bpm, filename, in_format, out_format):
        # randomly transform n times
        for n in range(self.transformations):

            # Random pitch
            semitones = random.randint(-4, 4)
            semitones_rand = (random.random()-.5) / 4

            # Pick random tempo in intervals of 10
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
            bpm_adjust = new_tempo / bpm
            bpm_rand = (random.random()-.5) / 250

            print('%dbpm %d/%d' % (new_tempo, n+1, self.transformations))

            # Make new tempo directory if necessary
            tempo_dir = ("./datasets/%s/%dbpm") % (self.dataset, new_tempo)
            if not path.exists(tempo_dir):
                os.mkdir(tempo_dir)

            # Use Sonic API to transform and save to tempo dir
            new_filename = re.sub('[()!@#$]', '-', filename)
            new_filename = new_filename.replace(
                self.dataset, self.dataset + "/%dbpm" % new_tempo).replace(
                ' ', '-').replace(
                in_format, '_%d%s' % (out_format))

            print("\tPitch change: %d  rand: %.6f" %
                  (semitones, semitones_rand))
            print("\tTempo change: %.2f  rand: %.6f" %
                  (bpm_adjust, bpm_rand))

            print('\tTransforming song with sox ...')
            tfm = sox.Transformer()
            tfm.pitch(semitones+semitones_rand)
            tfm.tempo(bpm_adjust+bpm_rand, "m")
            tfm.convert(samplerate=8192, channels=1)
            tfm.build(filename, new_filename)
            # self.bpm_pitch_adjust(
            #     filename,
            #     new_filename,
            #     bpm_adjust+bpm_rand,
            #     semitones+semitones_rand)
            # print("\tSaved to %s" % new_filename)
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

    def slice_songs_by_bars(self, save_rate=1, in_format='wav', out_format='wav'):

        total_songs = len(glob.glob("datasets/%s/%sbpm/*.%s" %
                                    (self.dataset, self.tempo, in_format)))

        print('\nBars: %d Tempo: %d Samples Per Bar: %d\n' %
              (self.bars, self.tempo, self.samples_per_bar))

        for filename in glob.glob("datasets/%s/%sbpm/*.%s" % (self.dataset, self.tempo, in_format)):

            print(filename)
            song = AudioSegment.from_file(filename, format=in_format)

            print("\tAnalyzing song ...")
            bpm, beats_per_bar, beats = self.tempo_analysis(filename)

            # If wrong BPM -> skip song
            if (round(bpm, -1) != self.tempo):
                print("\tError: %d is the wrong BPM" % int(bpm))
                continue

            print("\tActual BPM: %.5f" % bpm)
            print("\tSlicing by %d bar(s)" % (self.bars))

            # Iterate over beat at bar intervals
            hop = beats_per_bar * self.bars * 2
            slice_start = 0

            for i, beat in enumerate(range(0, len(beats) - hop, hop)):

                # if curr beat tempo is close to desired tempo
                if round(beats[beat]['bpm'], -1) != self.tempo:
                    print('\tError: Beat at %dbpm not %dbpm' %
                          (int(beats[beat]['bpm']), self.tempo))
                    continue

                # if slice start not on beat -> move to beat
                if round(slice_start, -1) != round(beats[beat]['time'] * 44100):
                    slice_start = round(beats[beat]['time'] * 44100)

                # slice length is num bars
                slice_stop = slice_start + self.samples_per_bar * self.bars
                try:
                    full_slice = song.get_sample_slice(
                        slice_start, slice_stop)
                except:
                    print('\tCould not slice bar from %d to %d' %
                            (slice_start, slice_stop))
                    break
                # start next slice at current stop
                slice_start = slice_stop

                # export slice
                try:
                    self.save_slices(i, save_rate, self.dataset, self.tempo, filename, beat, full_slice)
                except:
                    print('\tError: Could not save slice')

    def load_slices(self, file_format='wav'):

        num_songs = len(glob.glob('./datasets/%s/%dbpm/slices/*.%s' %
                                  (self.dataset, self.tempo, file_format)))

        print('loading %d songs ...' % num_songs)

        songs = np.zeros((num_songs,) + self.shape)
        len_songs = 0

        for i, filepath in enumerate(glob.iglob('./datasets/%s/%dbpm/slices/*.%s' % (self.dataset, self.tempo, file_format))):
            if i % 50 == 0:
                print()
            try:
                song = AudioSegment.from_file(filepath, format=file_format)
                songs[len_songs] = np.reshape(
                    np.array(song.get_array_of_samples()), self.shape)
                len_songs += 1
                print(".", end='')
            except:
                print("e", end='')
        print()

        return songs[:len_songs]

    def start_songs_at_first_beat(self):

        for filename in glob.glob("datasets/%s/*.mp3" % self.dataset):
            print(os.path.basename(filename))
            print("\tAnalyzing song ...")
            song = AudioSegment.from_mp3(filename)
            bpm, beats_per_bar, beats = self.tempo_analysis(filename)

            samples_per_beat = int(60 / bpm * 44100 * 4)
            start_beat = None

            # User input needed for determining first beat of song
            for beat in range(beats_per_bar*2):
                response = "replay"

                slice_start = round(beats[beat]['time'] * 44100)
                slice_stop = slice_start + samples_per_beat
                bar_slice = song.get_sample_slice(slice_start, slice_stop)
                input('\tPlaying slice %d. Press any key to continue\n' %
                      (beat + 1))

                while response.lower().strip() == 'replay':
                    play(bar_slice + bar_slice)
                    response = input(
                        "\tIs this loop correct? y/n/replay/skip\n")
                if response.lower().strip() == 'skip':
                    break
                elif response.lower().strip() in ['y', 'yes']:
                    start_beat = beat
                    break

            # Remove file
            os.remove(filename)

            # Replace file if given start beat
            if type(start_beat) is int:
                song = song[int(beats[start_beat]['time']*1000)                            :int(song.duration_seconds*1000)]
                wav_filename = filename.replace('.mp3', '.wav')
                song.export(
                    out_f=wav_filename,
                    format='wav',
                    tags={'tempo': bpm})
                print('\tMade wav %s' % os.path.basename(wav_filename))
            else:
                print('\tRemoved %s' % os.path.basename(filename))

    
    ## API Functions
    #################
    def sonic_api(self, filename, endpoint, params={'format': 'json'}):
        _, extension = os.path.splitext(filename)

        def set_params():
            return {
                'access_id': self.auth_codes[self.sonic_auth_index],
            }.update(params)
        def set_files():
            return {
                'input_file': ('song.%s' % extension, open(filename, 'rb'), "multipart/form-data")
            }
        
        response = requests.post(
            'https://api.sonicAPI.com/%s' % endpoint, files=set_files(), params=set_params())
        
        # Increment auth codes if necessary
        while response.status_code in [400, 401, 403]:
            self.sonic_auth_index += 1
            
            print('\tChanged Authorization: %d/%d' %
              (self.sonic_auth_index+1, len(self.auth_codes)))
            print('\tTrying Again ...')

            response = requests.post(
                'https://api.sonicAPI.com/%s', files=set_files(), params=set_params())
        
        # return content or raise exception
        if response.status_code == 200:
            return response.content
        else:
            print('\tStatus Code: %d' % response.status_code)
            raise Exception("Bad request bro")
    
    def tempo_analysis(self, filename):
        
        content = json.loads(self.sonic_api(
            filename, 'analyze/tempo'))['auftakt_result']
        beats = content['click_marks']
        beats_per_bar = content['clicks_per_bar']
        bpm = round(content['overall_tempo'])

        return bpm, beats_per_bar, beats

    def bpm_pitch_adjust(self, filename, new_filename, bpm_adjust, semitones):
        _, extension = os.path.splitext(filename)
        new_song = self.sonic_api(filename, 'process/elastique', params={
            'pitch_semitones': semitones,
            'tempo_factor': bpm_adjust,
            'format': extension,
        })

        with open(new_filename, mode='bx') as f:
            f.write(new_song)

        print("\tSaved as: %s" % new_filename)

    
    ## Helper Functions
    #################
    def save_slices(self, i, save_rate, filename, beat, bar_slice):
        if i % save_rate == 0:
            if not os.path.exists("datasets/%s/%dbpm/slices" % (self.dataset, self.tempo)):
                os.mkdir("datasets/%s/%dbpm/slices" %
                         (self.dataset, self.tempo))

            slice_export_path = "datasets/%s/%dbpm/slices/%s" % (
                self.dataset, self.tempo, os.path.basename(filename).replace(".wav", "_slice%d.wav" % beat))
            bar_slice.export(slice_export_path, format="wav")
            print('s', end='')

if __name__ == "__main__":
    ag = AudioGod('fkn_bs', 110, 1)
    # ag.preprocess()
    ag.load_slices()
