import torch

from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput

from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio


train_df, test_df = load_metadata()

sample = test_df.iloc[0]

y, sr = load_audio(sample["filepath"])

# 先测试 1 秒
y = crop_audio(
    y,
    sr,
    duration=1,
    seed=42,
)

waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

converter = WaveformToInput()

x = converter(
    waveform,
    sr,
)

print("Input tensor shape:", x.shape)

model = yamnet.yamnet(
    pretrained=True
)

model.eval()

with torch.no_grad():
    embeddings, logits = model(x)

print("Embedding shape:", embeddings.shape)
print("Logits shape:", logits.shape)