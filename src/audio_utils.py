import numpy as np
import librosa


TARGET_SR = 16000


def load_audio(filepath):
    """
    Load audio as mono and resample to 16 kHz.
    """
    y, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    return y, sr


def crop_audio(y, sr, duration, seed=42):
    """
    Randomly crop an audio signal to the required duration.

    duration:
        1, 2, or 5 seconds

    seed:
        fixed random seed for reproducibility
    """
    target_samples = int(duration * sr)

    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
        return y

    if len(y) == target_samples:
        return y

    rng = np.random.default_rng(seed)

    max_start = len(y) - target_samples
    start = rng.integers(0, max_start + 1)

    return y[start:start + target_samples]