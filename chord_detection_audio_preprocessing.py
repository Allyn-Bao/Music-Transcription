import os
import matplotlib
matplotlib.use('TkAgg')  # or 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd


def preprocess_audio_file_for_chords_detection(file_path, csv_path):
    """
    saves a .csv chroma spectrum of pitch class with timestamp given an .mp3 file
    :param file_path: str, path to .mp3 audio file
    :return: None
    """
    # load audio file
    # y: array of sampled amplitudes
    # sr: sample rate
    y, sr = librosa.load(file_path)

    # simple approach
    # "chroma" constant Q transform
    hop_length, bar_interval, bpm = get_hop_length(y, sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # potentially better approach for songs with drums / other percussive elements
    # Harmonic-Percussive Chroma - separates percussive elements from music
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    chroma_harm = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)

    chroma_df = pd.DataFrame(chroma_cqt.T)

    # add timestamp column
    timestamp_column = np.arange(chroma_cqt.shape[1]) * hop_length / sr
    chroma_df.insert(0, 'Timestamp', timestamp_column)

    # add file path column
    # Create a column with the file path only in the first row
    file_column = [file_path] + [None] * (chroma_df.shape[0] - 1)
    chroma_df.insert(0, 'File Path', file_column)

    # Save the DataFrame to a CSV file
    chroma_df.to_csv(csv_path, index=False, header=False)
    print(f"Chroma features saved to {csv_path}")

    print('hop_length = {}, bar_interval = {}, bpm = {}'.format(hop_length, bar_interval, bpm))
    print(chroma_df.head())


def get_hop_length(y, sr):
    """
    hope length = length / interval in # of samples where chroma is applied
    we arbitrarily decide that the hope_length is that of one bar of music assume 4/4
    (This feature may not be super reliable as it guesses the bpm of the song)
    :param y: array of sampled amplitudes
    :param sr: sample rate
    :return: hope length in # of samples
    """
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(bpm[0] / 2)   # I found that the default bpm detected is too fast which might not be necessary
    # assume 4 beats per bar
    bar_interval = 4 * 60 / bpm     # time interval of each bar in seconds
    num_samples_in_a_bar = bar_interval * sr
    return int(num_samples_in_a_bar), bar_interval, bpm


if __name__ == "__main__":
    path = os.path.join("musicFiles", "dry-muted-electric-guitar-chords-thunder-rush_92bpm_G_major.mp3")
    export_path = os.path.join("exported_csv", "dry-muted-electric-guitar-chords-thunder-rush_92bpm_G_major.csv")
    preprocess_audio_file_for_chords_detection(path, export_path)
