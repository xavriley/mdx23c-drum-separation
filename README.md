# MDX23C Drum Separation

A pip installable package for drum separation using the MDX23C model.

## Installation

### From source
```bash
git clone https://github.com/xavriley/mdx23c-drum-separation.git
cd mdx23c-drum-separation
pip install -e .
```

## Usage

### Command Line Interface

Basic usage:
```bash
mdx23c-separate --input_folder /path/to/audio/files --store_dir /path/to/output
```

With model checkpoint:
```bash
mdx23c-separate --input_folder /path/to/audio --store_dir /path/to/output --checkpoint /path/to/model.ckpt
```

Full options:
```bash
mdx23c-separate \
    --input_folder /path/to/audio \
    --store_dir /path/to/output \
    --checkpoint /path/to/model.ckpt \
    --config /path/to/config.yaml \
    --use_tta \
    --extract_instrumental \
    --normalize
```

### Python API

#### Convenience Methods (Recommended)

```python
from mdx23c import demix_kit_from_mix, demix_stems_from_kit

# Separate full mix into drum kit components
# Uses pre-trained model, downloads automatically (~500MB)
kit_stems = demix_kit_from_mix('song.wav', output_dir='output/')

# Separate drum kit into individual stems 
# Uses pre-trained model, downloads automatically (~500MB)
drum_stems = demix_stems_from_kit('drums.wav', output_dir='output/')

# With additional options
stems = demix_kit_from_mix(
    'song.wav',
    use_tta=True,          # Better quality, slower
    normalize=True,        # Normalize audio
    sample_rate=48000,     # Custom sample rate
    device='cuda'          # Force specific device
)
```

#### General API

```python
from mdx23c import demix_audio, load_model

# High-level API - separate drums from an audio file
stems = demix_audio(
    'path/to/audio.wav',
    output_dir='path/to/output',
    checkpoint_path='path/to/model.ckpt',
    use_tta=True
)

# Lower-level API - load model and process manually
model, config, device = load_model('path/to/model.ckpt')
# ... use model for custom processing

# Use config and weights directly (no file paths needed)
stems = demix_audio(
    'audio.wav',
    config=my_config_object,
    state_dict=my_model_weights
)
```

#### Model Cache Management

```python
from mdx23c import list_cached_models, clear_model_cache, get_model_info

# List downloaded models and their sizes
list_cached_models()

# Get info about available models
info = get_model_info('kit_from_mix')

# Clear cached models to free disk space
clear_model_cache('kit_from_mix')  # Clear specific model
clear_model_cache()                # Clear all models
```

### Environment Variables

You can set environment variables to avoid specifying paths repeatedly:

```bash
export MDX23C_CKPT=/path/to/your/model.ckpt
export MDX23C_CONFIG=/path/to/your/config.yaml
```

## Model Checkpoints

This package provides the model architecture but you need to provide trained weights. You can:

1. Train your own model using the original MDX23C training code
2. Use pre-trained weights if available from the original authors
3. Set the checkpoint path via `--checkpoint` argument or `MDX23C_CKPT` environment variable

## Configuration

The package includes default configuration that works for most cases. You can:

1. Use the built-in defaults (no config file needed)
2. Provide a custom YAML config file via `--config` or `MDX23C_CONFIG` environment variable
3. Override specific settings via command line arguments

## Requirements

- Python 3.8+
- PyTorch 2.0.1+
- See `requirements.txt` for full dependencies

## License

MIT License (update as needed)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

Based on the MDX23C model architecture for music source separation.
