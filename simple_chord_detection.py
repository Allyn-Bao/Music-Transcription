from pydub import AudioSegment
import librosa
import os
import matplotlib
matplotlib.use('TkAgg')  # or 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt


def chord_detection(mp3_path):
    # Using librosa
    # y: numpy array of amplitude
    # sr: sample rate
    y, sr = librosa.load(mp3_path, sr=22050, mono=True)

    print('y: {}'.format(y))
    print('sr: {}'.format(sr))
    plot_waveform(y, sr)


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


def bar_interval_from_bpm(bpm):
    """
    calculate the time interval it takes to complete one bar, assume 4/4
    :param bpm: beats per minutes
    :return: time interval in seconds
    """
    bar_interval = 4 * 60 / bpm     # 4 beats per minute
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
