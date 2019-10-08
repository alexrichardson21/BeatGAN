import glob
import copy
import io
import json
import os
import random
import re
import sys
import time
import urllib
from os import path

# import ffmpy
import matplotlib.pyplot as plt
import numpy as np
import requests
# import soundfile as sf
import sox
# from aubio import notes, onset, pitch, source, tempo
from pydub import AudioSegment, silence
from pydub.playback import play
# from scipy.io import wavfile

# Preprocessing Order:
# 1. Soundcloud Playlist Download
# 2. Random Transformations and Bin Tempos
# 3. Slice Songs at N bar Intervals
# 4. Load slices


class AudioGod():
    def __init__(self, dataset, tempo, bars, shape):

        self.transformations = 10

        self.channels = 1
        self.bit_rate = 128 * 1000
        self.bit_depth = 16
        self.sample_rate = 44100 #self.bit_rate // self.bit_depth

        self.dataset = dataset
        self.tempo = tempo
        self.bars = bars
        self.shape = shape
        self.samples_per_bar = self.sample_rate * 60 // self.tempo * 4

        self.sonic_auth_index = 0
        with open('auth.json') as f:
            self.auth_codes = json.load(f)

    # Main Functions
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
                basefilename = os.path.basename(filename).replace('.%s' % in_format, '')
                print(basefilename)
                
                # Checks to see if song has already been transformed
                files = glob.glob("datasets/%s/%sbpm/*.%s" %
                                  (self.dataset, self.tempo, out_format))
                if ''.join(files).find(basefilename) != -1:
                    print("\tFile already transformed")
                    continue

                song = AudioSegment.from_file(filename, format=in_format)

                # If song longer than 5 minutes -> skip song
                if song.duration_seconds > 60 * 5:
                    print('\tError: Song longer than 5 minutes')
                    continue

                print('\tAnalyzing song ...')
                bpm, beats_per_bar, beats = self.tempo_analysis(filename)
                print('\tSong tempo: %d' % int(bpm))

                if abs(bpm - self.tempo) > 35:
                    print('\tError: Song tempo too far from desired tempo')
                    continue

                # randomly transform n times
                self.transform(bpm, beats_per_bar, beats,
                               filename, in_format, out_format)

    def transform(self, bpm, beats_per_bar, beats, filename, in_format, out_format, randomize_tempo=False):
        # randomly transform n times
        for trans in range(self.transformations):
            # Random pitch
            semitones_rand = (random.random()-.5) * 6

            # Random tempo
            bpm_adjust = self.tempo / bpm
            bpm_rand = (random.random()-.5) / 200

            print('%d/%d' %
                  (trans+1, self.transformations))

            # Make new tempo directory if necessary
            tempo_dir = ("datasets/%s/%dbpm") % (self.dataset, self.tempo)
            if not path.exists(tempo_dir):
                os.mkdir(tempo_dir)

            # Change filename
            _, extension = os.path.splitext(filename)
            basename = os.path.basename(filename).replace(extension, "")
            new_filename = ("datasets/%s/%dbpm/%s_%d.%s" %
                            (self.dataset,
                             self.tempo,
                             basename,
                             trans+1,
                             out_format))

            # Use sox to transform and save to tempo dir
            print("\tPitch change: %.6f" %
                  (semitones_rand))
            print("\tTempo change: %.2f  rand: %.6f" %
                  (bpm_adjust, bpm_rand))
            print('\tTransforming song with sox ...')
            
            tfm = sox.Transformer()
            tfm.speed(bpm_adjust+bpm_rand)
            tfm.pitch(semitones_rand)
            tfm.convert(samplerate=self.sample_rate, n_channels=self.channels, bitdepth=self.bit_depth)
            tfm.build(filename, new_filename)

            # print("\tOutputting song in desired format ...")
            
            
            # song = AudioSegment.from_file(new_filename, format=out_format)
            # song.export(new_filename,
            #             format=out_format,)
                        # bitrate="%dk" % (self.bit_rate//1000))

            print("\tUploading metadata ...")

            new_bpm = bpm * (bpm_adjust + bpm_rand)
            new_beats = copy.deepcopy(beats)
            for beat in new_beats:
                beat['time'] *= (bpm_adjust + bpm_rand)
                beat['bpm'] *= (bpm_adjust + bpm_rand)
            data = {
                1: [
                    new_bpm,
                    beats_per_bar,
                    new_beats,
                ]
            }

            # Update metadata json
            # meta_json_path = "datasets/%s/%sbpm/%s_metadata.json" % (self.dataset, self.tempo, basename)

            meta_filename = ("datasets/%s/%dbpm/%s_%d_metadata.json" %
                            (self.dataset,
                             self.tempo,
                             basename,
                             trans+1))

            # if not os.path.isfile(meta_json_path):
            #     with open(meta_json_path, 'w') as f:
            #         json.dump({}, f)

            # with open(meta_json_path, 'r') as f:
            #     metadata = json.load(f)

            # metadata.update(data)

            with open(meta_filename, 'w') as f:
                json.dump(data, f)

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

        # total_songs = len(glob.glob("datasets/%s/%sbpm/*.%s" %
        #                             (self.dataset, self.tempo, in_format)))

        print('\nBars: %d Tempo: %d Samples Per Bar: %d\n' %
              (self.bars, self.tempo, self.samples_per_bar))

        for filename in glob.glob("datasets/%s/%sbpm/*.%s" % (self.dataset, self.tempo, in_format)):

            print(filename)
            song = AudioSegment.from_file(filename, format=in_format)

            # Find song metadata in json file
            _, extension = os.path.splitext(filename)
            basename = os.path.basename(filename).replace(extension, "")

            with open("./datasets/%s/%dbpm/%s_metadata.json" %
                      (self.dataset, self.tempo, basename)) as f:
                metadata = json.load(f)

            bpm, beats_per_bar, beats = tuple(metadata['1'])

            # If wrong BPM -> skip song
            if (round(bpm, -1) != self.tempo):
                print("\tError: %d is the wrong BPM" % int(bpm))
                continue

            print("\tActual BPM: %.5f" % bpm)
            print("\tSlicing by %d bar(s)" % (self.bars))

            # Iterate over beat at bar intervals
            hop = beats_per_bar * self.bars * 4
            slice_start = 0

            for i, beat in enumerate(range(0, len(beats) - hop, hop)):

                # if curr beat tempo is close to desired tempo
                if round(beats[beat]['bpm'], -1) != self.tempo:
                    print('\tError: Beat at %dbpm not %dbpm' %
                          (int(beats[beat]['bpm']), self.tempo))
                    continue

                # if slice start not on beat -> move to beat
                # if round(slice_start, -1) != round(beats[beat]['time'] * self.sample_rate):
                #     slice_start = round(beats[beat]['time'] * self.sample_rate)

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
                    self.save_slices(i, save_rate, filename, beat, full_slice)
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

    # DEPRECATED
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
                song = song[int(beats[start_beat]['time']*1000)
                                :int(song.duration_seconds*1000)]
                wav_filename = filename.replace('.mp3', '.wav')
                song.export(
                    out_f=wav_filename,
                    format='wav',
                    tags={'tempo': bpm})
                print('\tMade wav %s' % os.path.basename(wav_filename))
            else:
                print('\tRemoved %s' % os.path.basename(filename))

    # API Functions
    #################

    def sonic_api(self, filename, endpoint, sonic_params={'format': 'json'}):
        _, extension = os.path.splitext(filename)

        def set_params():
            s_params = {
                'access_id': self.auth_codes[self.sonic_auth_index],
                'format': 'json'
            }  # .update(sonic_params)
            return s_params

        def set_files():
            return {
                'input_file': ('song%s' % extension, open(filename, 'rb'), "multipart/form-data")
            }

        response = requests.post(
            'https://api.sonicAPI.com/%s' % endpoint, files=set_files(), params=set_params())
        # print(response.url)

        # Increment auth codes if necessary
        while response.status_code in [400, 401, 403]:
            print('\tStatus Code: %d' % response.status_code)
            time.sleep(2)
            
            self.sonic_auth_index += 1
            if self.sonic_auth_index >= len(self.auth_codes):
                raise Exception("No more usable access id's")

            print('\tChanged Authorization: %d/%d' %
                  (self.sonic_auth_index+1, len(self.auth_codes)))
            print('\tTrying Again ...')

            response = requests.post(
                'https://api.sonicAPI.com/%s' % endpoint, files=set_files(), data=set_params())

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

    # Helper Functions
    #################

    def save_slices(self, i, save_rate, filename, beat, bar_slice):
        if i % save_rate == 0:
            if not os.path.exists("datasets/%s/%dbpm/slices" % (self.dataset, self.tempo)):
                os.mkdir("datasets/%s/%dbpm/slices" %
                         (self.dataset, self.tempo))

            slice_export_path = "datasets/%s/%dbpm/slices/%s" % (
                self.dataset, self.tempo, os.path.basename(filename).replace(".wav", "_slice%d.wav" % i))
            bar_slice.export(slice_export_path, format="wav")
            print('s', end='')


if __name__ == "__main__":
    ag = AudioGod('yung_gan', 120, 1, (210, 1, 420))
    ag.slice_songs_by_bars()
    # ag.preprocess()
    # ag.load_slices()
