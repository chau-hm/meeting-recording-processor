"""ASR backend adapters."""

from .base import AsrBackend, ProgressCallback
from .qwen3 import Qwen3Backend
from .sensevoice import SenseVoiceBackend
from .vibevoice import VibeVoiceBackend

__all__ = [
    "AsrBackend",
    "ProgressCallback",
    "Qwen3Backend",
    "SenseVoiceBackend",
    "VibeVoiceBackend",
]
