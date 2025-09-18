"""Model hub utilities for downloading pre-trained models."""

import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any
import torch

# Model configurations
MODEL_CONFIGS = {
    'kit_from_mix': {
        'repo_id': 'xavriley/source_separation_mirror',
        'checkpoint_file': 'model_mdx23c_ep_168_sdr_7.0207.ckpt',
        'config_file': 'config_musdb18_mdx23c.yaml.1',
        'description': 'MDX23C model for separating full mix into drum kit components'
    },
    'stems_from_kit': {
        'repo_id': 'xavriley/source_separation_mirror', 
        'checkpoint_file': 'drumsep_5stems_mdx23c_jarredou.ckpt',
        'config_file': 'config_mdx23c_drumsep2025.yaml',
        'description': 'MDX23C model for separating drum kit into individual stems'
    }
}


def get_model_cache_dir() -> Path:
    """Get the directory for caching downloaded models."""
    # Use XDG cache directory or fallback to user's home
    cache_dir = os.environ.get('XDG_CACHE_HOME')
    if cache_dir:
        cache_dir = Path(cache_dir)
    else:
        cache_dir = Path.home() / '.cache'
    
    model_cache = cache_dir / 'mdx23c_models'
    model_cache.mkdir(parents=True, exist_ok=True)
    return model_cache


def download_model_files(model_key: str, force_download: bool = False) -> Tuple[str, str]:
    """
    Download model checkpoint and config files from Hugging Face.
    
    Args:
        model_key: Key identifying the model ('kit_from_mix' or 'stems_from_kit')
        force_download: Whether to re-download even if files exist
        
    Returns:
        Tuple of (checkpoint_path, config_path)
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    cache_dir = get_model_cache_dir()
    model_dir = cache_dir / model_key
    model_dir.mkdir(exist_ok=True)
    
    checkpoint_path = model_dir / config['checkpoint_file']
    config_path = model_dir / config['config_file']
    
    # Check if files already exist
    if not force_download and checkpoint_path.exists() and config_path.exists():
        print(f"Using cached model files for {model_key}")
        return str(checkpoint_path), str(config_path)
    
    # Import huggingface_hub here to avoid import errors if not installed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for automatic model downloading. "
            "Install it with: pip install huggingface_hub"
        )
    
    print(f"\n🔄 Downloading {config['description']}")
    print(f"⚠️  Warning: Model files are approximately 500MB in size")
    print(f"📁 Cache directory: {model_dir}")
    
    try:
        # Download checkpoint
        print(f"Downloading checkpoint: {config['checkpoint_file']}")
        downloaded_checkpoint = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['checkpoint_file'],
            cache_dir=str(cache_dir),
            force_download=force_download
        )
        
        # Download config  
        print(f"Downloading config: {config['config_file']}")
        downloaded_config = hf_hub_download(
            repo_id=config['repo_id'], 
            filename=config['config_file'],
            cache_dir=str(cache_dir),
            force_download=force_download
        )
        
        # Copy to our model directory structure for easier management
        import shutil
        shutil.copy2(downloaded_checkpoint, checkpoint_path)
        shutil.copy2(downloaded_config, config_path)
        
        print(f"✅ Downloaded and cached {model_key} model files")
        return str(checkpoint_path), str(config_path)
        
    except Exception as e:
        raise RuntimeError(f"Failed to download model files for {model_key}: {e}")


def load_model_from_hub(model_key: str, device: str = None, force_download: bool = False):
    """
    Load a pre-trained model from the hub.
    
    Args:
        model_key: Key identifying the model ('kit_from_mix' or 'stems_from_kit')
        device: Device to load model on (auto-detected if None)
        force_download: Whether to re-download even if files exist
        
    Returns:
        Tuple of (model, config, device, checkpoint_path, config_path)
    """
    from .inference import load_model
    
    # Download model files
    checkpoint_path, config_path = download_model_files(model_key, force_download)
    
    # Load model using existing infrastructure
    model, config, device = load_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device
    )
    
    return model, config, device, checkpoint_path, config_path


def get_model_info(model_key: str = None) -> Dict[str, Any]:
    """
    Get information about available models.
    
    Args:
        model_key: Specific model to get info for, or None for all models
        
    Returns:
        Dictionary with model information
    """
    if model_key is None:
        return MODEL_CONFIGS
    elif model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]
    else:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")


def clear_model_cache(model_key: str = None):
    """
    Clear cached model files.
    
    Args:
        model_key: Specific model to clear, or None to clear all
    """
    cache_dir = get_model_cache_dir()
    
    if model_key is None:
        # Clear all cached models
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Cleared all cached models from {cache_dir}")
    else:
        # Clear specific model
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}")
        
        model_dir = cache_dir / model_key
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            print(f"Cleared cached files for {model_key}")
        else:
            print(f"No cached files found for {model_key}")


def list_cached_models():
    """List all cached models and their sizes."""
    cache_dir = get_model_cache_dir()
    
    if not cache_dir.exists():
        print("No model cache directory found")
        return
    
    print(f"Model cache directory: {cache_dir}")
    total_size = 0
    
    for model_key in MODEL_CONFIGS.keys():
        model_dir = cache_dir / model_key
        if model_dir.exists():
            size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            total_size += size
            print(f"  {model_key}: {size / (1024**2):.1f} MB")
        else:
            print(f"  {model_key}: Not cached")
    
    print(f"Total cache size: {total_size / (1024**2):.1f} MB")
