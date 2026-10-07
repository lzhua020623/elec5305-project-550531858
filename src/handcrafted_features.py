import numpy as np
import librosa


def summarize_feature(feature):
    """
    Convert a frame-level feature into observation-level statistics.
    """
    feature = np.asarray(feature).flatten()

    return [
        np.mean(feature),
        np.std(feature),
    ]


def extract_handcrafted_features(y, sr):
    """
    Extract interpretable handcrafted audio features from one audio observation.
    """

    features = []

    # 1. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features.extend(summarize_feature(zcr))

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)[0]

    # Use temporal energy behaviour rather than absolute energy only
    features.extend(summarize_feature(rms))

    energy_variability = np.std(rms) / (np.mean(rms) + 1e-8)
    features.append(energy_variability)

    # 3. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend(summarize_feature(centroid))

    # 4. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend(summarize_feature(bandwidth))

    # 5. Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend(summarize_feature(rolloff))

    # 6. Spectral Flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.extend(summarize_feature(flatness))

    # 7. Spectral Flux
    stft = np.abs(librosa.stft(y))
    diff = np.diff(stft, axis=1)

    spectral_flux = np.sqrt(np.sum(diff ** 2, axis=0))
    features.extend(summarize_feature(spectral_flux))

    # 8. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for coefficient in mfcc:
        features.extend(summarize_feature(coefficient))

    return np.array(features, dtype=np.float32)