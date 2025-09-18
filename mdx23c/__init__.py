"""MDX23C Drum Separation Package

A pip installable package for drum separation using the MDX23C model.
"""

__version__ = "0.1.0"
__author__ = "MDX23C Drum Separation"
__email__ = ""

from .inference import demix_audio, load_model, demix_kit_from_mix, demix_stems_from_kit
from .models import TFC_TDF_net
from .config import build_default_mdx23c_config
from .model_hub import get_model_info, clear_model_cache, list_cached_models

__all__ = [
    "demix_audio",
    "load_model", 
    "demix_kit_from_mix",
    "demix_stems_from_kit",
    "TFC_TDF_net",
    "build_default_mdx23c_config",
    "get_model_info",
    "clear_model_cache", 
    "list_cached_models"
]
