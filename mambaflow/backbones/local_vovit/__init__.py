"""Small local subset of the gesture-guided/MambaVoice feature utilities.

This package avoids depending on an external gesture-guided repo when the flow
model only needs the MambaVoice-style audio and visual feature embeddings.
"""

from .feature_extractor import LocalMambaVoiceFeatureExtractor

__all__ = ["LocalMambaVoiceFeatureExtractor"]
