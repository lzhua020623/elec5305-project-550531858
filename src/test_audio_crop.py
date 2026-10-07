from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio


train_df, test_df = load_metadata()

sample = test_df.iloc[0]

filepath = sample["filepath"]

print("File:", filepath)
print("Category:", sample["category"])

y, sr = load_audio(filepath)

print("\nOriginal:")
print("Sample rate:", sr)
print("Samples:", len(y))
print("Duration:", len(y) / sr)

for duration in [1, 2, 5]:
    cropped = crop_audio(
        y,
        sr,
        duration=duration,
        seed=42
    )

    print(f"\n{duration}s observation:")
    print("Samples:", len(cropped))
    print("Duration:", len(cropped) / sr)