#pip install pydub

import os
import librosa
import numpy as np
import tensorflow as tf

# Set constants for the preprocessing
SAMPLE_RATE = 16000
DURATION = 1  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """Load an audio file and resample it to the given sample rate."""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    # Ensure consistency in audio length
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    elif len(audio) < SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(audio)
        offset = padding // 2
        audio = np.pad(audio, (offset, SAMPLES_PER_TRACK - len(audio) - offset), 'constant')
    return audio

def extract_features(audio):
    """Extract Mel-spectrogram features from the audio."""
    spectrogram = librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

def normalize(spectrogram):
    """Normalize the spectrogram."""
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    return (spectrogram - mean) / std
def prepare_dataset(file_paths):
    """Prepare the dataset by processing and normalizing the audio files."""
    # Load and process the audio files
    audios = [load_audio(fp) for fp in file_paths]
    # Extract features
    features = [extract_features(audio) for audio in audios]
    # Normalize features
    normalized_features = [normalize(f) for f in features]

    return normalized_features

def create_tf_dataset(file_paths, labels):
    """Create a TensorFlow dataset from the given file paths and labels."""
    features = prepare_dataset(file_paths)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
# Example file paths and labels
file_paths = ['path/to/audio1.wav', 'path/to/audio2.wav', ...]
labels = [0, 1, ...]  # Corresponding labels for the audio files

# Create dataset
dataset = create_tf_dataset(file_paths, labels)

# Iterate over the dataset
for features, label in dataset.take(1):
    print("Features shape:", features.shape)
    print("Label:", label.numpy())
