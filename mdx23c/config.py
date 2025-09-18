"""Configuration utilities for MDX23C."""

import os
from types import SimpleNamespace


def _dict_to_namespace(d: dict):
    """Convert nested dictionary to nested SimpleNamespace."""
    def convert(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: convert(v) for k, v in obj.items()})
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert(v) for v in obj)
        else:
            return obj
    return convert(d)


def build_default_mdx23c_config():
    """Build default configuration for MDX23C model."""
    cfg = {
        "audio": {
            "chunk_size": 523776,
            "dim_f": 1024,
            "dim_t": 1024,
            "hop_length": 512,
            "n_fft": 2048,
            "num_channels": 2,
            "sample_rate": 44100,
            "min_mean_abs": 0.000,
        },
        "model": {
            "act": "gelu",
            "bottleneck_factor": 4,
            "growth": 128,
            "norm": "InstanceNorm",
            "num_blocks_per_scale": 2,
            "num_channels": 128,
            "num_scales": 5,
            "num_subbands": 4,
            "scale": [2, 2],
        },
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "grad_clip": 0,
            "instruments": ["kick", "snare", "toms", "hh", "cymbals"],
            "target_instrument": None,
            "use_amp": True,
        },
        "inference": {
            "batch_size": 2,
            "dim_t": 512,
            "num_overlap": 4,
            "normalize": False,
        },
    }
    return _dict_to_namespace(cfg)


def maybe_load_config_from_yaml():
    """Try to load configuration from YAML file specified in MDX23C_CONFIG env var."""
    cfg_path = os.environ.get('MDX23C_CONFIG', '').strip()
    if not cfg_path:
        return None
    try:
        import yaml
        with open(cfg_path, 'r') as f:
            data = yaml.safe_load(f)
        return _dict_to_namespace(data)
    except Exception as e:
        print(f"Warning: failed to load MDX23C_CONFIG {cfg_path}: {e}")
        return None


def prefer_target_instrument(config) -> list:
    """Get target instruments from config, preferring target_instrument if set."""
    target = getattr(getattr(config, 'training', SimpleNamespace()), 'target_instrument', None)
    if target:
        return [target]
    return list(getattr(config.training, 'instruments', []))
