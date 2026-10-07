from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio
from handcrafted_features import extract_handcrafted_features


train_df, test_df = load_metadata()

sample = test_df.iloc[0]

y, sr = load_audio(sample["filepath"])

for duration in [1, 2, 5]:

    cropped = crop_audio(
        y,
        sr,
        duration=duration,
        seed=42
    )

    features = extract_handcrafted_features(cropped, sr)

    print(f"{duration}s:")
    print("Feature vector shape:", features.shape)
    print("Contains NaN:", bool((features != features).any()))
    print("First 10 values:", features[:10])
    print()