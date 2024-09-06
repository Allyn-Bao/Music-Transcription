"""
This File is depreciated as a better approach is found, although some functions might still be helpful
check chord_detection_audio_preprocessing.py
"""

from pydub import AudioSegment
import librosa
import os
import matplotlib
matplotlib.use('TkAgg')  # or 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np


def chord_detection(mp3_path):
    # Using librosa to load audio files as an numpy array
    # y: numpy array of amplitude
    # sr: sample rate
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)

    # log array and display the waveform
    print('y: {}'.format(y))
    print('sr: {}'.format(sr))
    plot_waveform(y, sr)

    # to determine the interval in which we detect the chords
    # estimate bpm
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(bpm[0])
    print("bpm: {}".format(bpm))
    # choose the interval to be the interval of a bar base on bpm
    num_samples_per_bar = int(get_num_samples_per_bar(bpm, sr))
    print("number of samples per bar: {}".format(num_samples_per_bar))

    # convert to frequency domain
    # segment music into bars
    for i in range(0, len(y), num_samples_per_bar):
        segment = y[i : i + num_samples_per_bar]

        # apply FFT
        freq_domain_segment = np.fft.fft(segment)
        magnitude_in_db = librosa.amplitude_to_db(np.abs(freq_domain_segment))
        frequencies = np.fft.fftfreq(len(segment), 1/sr)
        plot_freq_domain(magnitude_in_db, frequencies, num_samples_per_bar, i)


"""
Helper Functions
"""
def plot_waveform(y, sr):
    """
    display the waveform give the output numpy array and sample rate
    :param y: numpy array of amplitude of each sample
    :param sr: sample rate
    :return: None
    """
    time = librosa.times_like(y, sr=sr)

    # Plot the waveform
    plt.figure()
    plt.plot(time, y)
    plt.xlabel('i-th sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


def plot_freq_domain(magnitude_in_db, frequencies, num_samples_per_bar, i):
    """
    plot the frequency domain of a segment i of the audio file
    :param freq_domain_segment:
    :param frequencies:
    :param num_samples_per_bar:
    :return: None
    """
    # Limit the frequencies to the range where typical musical notes lie (20 Hz to 5000 Hz)
    idx = np.where((frequencies >= 20) & (frequencies <= 5000))
    plt.figure()
    plt.semilogx(frequencies[idx], np.abs(magnitude_in_db)[idx])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT of segment {i // num_samples_per_bar + 1}')
    plt.xlim(20, 5000)  # Limit the x-axis to the range of interest
    plt.show()


def bar_interval_from_bpm(bpm):
    """
    calculate the time interval it takes to complete one bar, assume 4/4
    :param bpm: beats per minutes
    :return: time interval in seconds
    """
    bar_interval = 4 * 60 / bpm
    return bar_interval


def get_num_samples_per_bar(bpm, sr):
    """
    return number of samples that completes each bar of the music file
    :param bpm: beats per minutes
    :param sr: sample rate
    :return: number of samples, integer
    """
    seconds_in_a_bar = bar_interval_from_bpm(bpm)
    num_samples = seconds_in_a_bar * sr
    return num_samples


if __name__ == "__main__":
    path = os.path.join("musicFiles", "dry-muted-electric-guitar-chords-thunder-rush_92bpm_G_major.mp3")
    chord_detection(path)
