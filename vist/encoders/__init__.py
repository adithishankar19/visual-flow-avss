"""Audio and visual encoders adapted from VoViT and MambaVoice.

The ST-GCN over facial landmarks follows VoViT / Acappella
(https://github.com/JuanFMontesinos/VoViT) and the band-split attention audio
encoder follows MambaVoice.
"""

from .feature_extractor import LocalMambaVoiceFeatureExtractor

__all__ = ["LocalMambaVoiceFeatureExtractor"]
