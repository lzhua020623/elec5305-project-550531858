from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio
from yamnet_features import YAMNetFeatureExtractor


train_df, test_df = load_metadata()

sample = test_df.iloc[0]

y, sr = load_audio(sample["filepath"])

extractor = YAMNetFeatureExtractor()

for duration in [1, 2, 5]:

    if duration < 5:
        audio = crop_audio(
            y,
            sr,
            duration=duration,
            seed=42
        )
    else:
        audio = y

    embedding = extractor.extract(
        audio,
        sr
    )

    print(f"{duration}s:")
    print("Embedding shape:", embedding.shape)
    print("Contains NaN:", bool((embedding != embedding).any()))
    print("Non-zero values:", (embedding != 0).sum())
    print("Mean:", embedding.mean())
    print("Norm:", (embedding ** 2).sum() ** 0.5)
    print("First 5 values:", embedding[:5])
    print()
    