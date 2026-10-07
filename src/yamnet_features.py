import numpy as np
import torch

from torch_vggish_yamnet import yamnet
from torch_vggish_yamnet.input_proc import WaveformToInput


class YAMNetFeatureExtractor:
    def __init__(self):
        self.converter = WaveformToInput()

        self.model = yamnet.yamnet(
            pretrained=True
        )

        self.model.eval()

    def extract(self, y, sr):
        """
        Extract one fixed 1024-dimensional YAMNet embedding
        from an audio observation.
        """

        waveform = torch.tensor(
            y,
            dtype=torch.float32
        ).unsqueeze(0)

        x = self.converter(
            waveform,
            sr
        )

        with torch.no_grad():
            embeddings, _ = self.model(x)

        # Shape may be:
        # [num_patches, 1024, 1, 1]
        embeddings = embeddings.squeeze(-1).squeeze(-1)

        # Average embeddings across all YAMNet patches
        embedding = embeddings.mean(dim=0)

        return embedding.cpu().numpy().astype(np.float32)