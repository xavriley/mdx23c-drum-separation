"""Inference utilities and main processing functions."""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from .config import prefer_target_instrument, maybe_load_config_from_yaml, build_default_mdx23c_config
from .models import TFC_TDF_net
from .audio_utils import normalize_audio, denormalize_audio


def _get_windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    """Create a windowing array for overlapping chunks."""
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(config, model: torch.nn.Module, mix: np.ndarray, device: str, model_type: str, pbar: bool = False):
    """
    Demix audio using the provided model.
    
    Args:
        config: Model configuration
        model: PyTorch model for source separation
        mix: Input audio as numpy array (channels, samples)
        device: Device to run inference on
        model_type: Type of model ('mdx23c' or other)
        pbar: Whether to show progress bar
        
    Returns:
        Dictionary mapping instrument names to separated audio arrays
    """
    mix = torch.tensor(mix, dtype=torch.float32)
    mode = 'generic' if model_type != 'htdemucs' else 'demucs'

    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = getattr(config.inference, 'chunk_size', getattr(config.audio, 'chunk_size', mix.shape[-1]))
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _get_windowing_array(chunk_size, fade_size)
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                pad_mode = "reflect" if (mode == "generic" and chunk_len > chunk_size // 2) else "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)
                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    if mode == "generic":
                        window = windowing_array.clone()
                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            if i - step == 0:
                                window[:fade_size] = 1
                            elif i >= mix.shape[1]:
                                window[-fade_size:] = 1
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0
                    batch_data.clear()
                    batch_locations.clear()
                if progress_bar:
                    progress_bar.update(step)
            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    instruments = config.training.instruments if mode == "demucs" else prefer_target_instrument(config)
    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}
    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def apply_tta(config, model: torch.nn.Module, mix: np.ndarray, waveforms_orig: dict, device: str, model_type: str):
    """Apply test-time augmentation to improve separation quality."""
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1
    return waveforms_orig


def load_model(checkpoint_path: str = None, 
                config_path: str = None, 
                config = None,
                state_dict = None,
                device: str = None):
    """
    Load a trained MDX23C model.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        config_path: Path to config YAML file (optional)
        config: Config object directly (overrides config_path and env vars)
        state_dict: Model state dict directly (overrides checkpoint_path and env vars)
        device: Device to load model on (auto-detected if None)
        
    Returns:
        Tuple of (model, config, device)
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Load config - prioritize direct config argument
    if config is not None:
        # Use provided config object directly
        pass
    elif config_path:
        # Load from specific path
        os.environ['MDX23C_CONFIG'] = config_path
        config = maybe_load_config_from_yaml()
        if config is None:
            raise ValueError(f"Failed to load config from {config_path}")
    else:
        # Try environment variable, then default
        config = maybe_load_config_from_yaml()
        if config is None:
            config = build_default_mdx23c_config()
    
    # Initialize model
    model = TFC_TDF_net(config)
    
    # Load weights - prioritize direct state_dict argument
    if state_dict is not None:
        # Use provided state dict directly
        try:
            load_res = model.load_state_dict(state_dict, strict=False)
            missing = getattr(load_res, 'missing_keys', [])
            unexpected = getattr(load_res, 'unexpected_keys', [])
            num_loaded = len([k for k in state_dict.keys() if k not in unexpected])
            total_params = len(list(model.state_dict().keys()))
            print(f"Loaded state dict | loaded {num_loaded}/{total_params} tensors")
            if missing:
                print(f"Warning: {len(missing)} missing keys (config mismatch likely). Example: {missing[:5]}")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys in state dict. Example: {unexpected[:5]}")
        except Exception as e:
            print(f"Warning: failed to load provided state dict: {e}")
    elif checkpoint_path:
        # Load from checkpoint file
        try:
            loaded_state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['state']
            if 'state_dict' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['state_dict']
            load_res = model.load_state_dict(loaded_state_dict, strict=False)
            missing = getattr(load_res, 'missing_keys', [])
            unexpected = getattr(load_res, 'unexpected_keys', [])
            num_loaded = len([k for k in loaded_state_dict.keys() if k not in unexpected])
            total_params = len(list(model.state_dict().keys()))
            print(f"Loaded checkpoint: {checkpoint_path} | loaded {num_loaded}/{total_params} tensors")
            if missing:
                print(f"Warning: {len(missing)} missing keys (config mismatch likely). Example: {missing[:5]}")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys in checkpoint. Example: {unexpected[:5]}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {checkpoint_path}: {e}")
    else:
        # Try loading from environment variable
        ckpt_path = os.environ.get('MDX23C_CKPT', '').strip()
        if ckpt_path:
            try:
                loaded_state_dict = torch.load(ckpt_path, map_location='cpu')
                if 'state' in loaded_state_dict:
                    loaded_state_dict = loaded_state_dict['state']
                if 'state_dict' in loaded_state_dict:
                    loaded_state_dict = loaded_state_dict['state_dict']
                load_res = model.load_state_dict(loaded_state_dict, strict=False)
                missing = getattr(load_res, 'missing_keys', [])
                unexpected = getattr(load_res, 'unexpected_keys', [])
                num_loaded = len([k for k in loaded_state_dict.keys() if k not in unexpected])
                total_params = len(list(model.state_dict().keys()))
                print(f"Loaded checkpoint from MDX23C_CKPT: {ckpt_path} | loaded {num_loaded}/{total_params} tensors")
                if missing:
                    print(f"Warning: {len(missing)} missing keys (config mismatch likely). Example: {missing[:5]}")
                if unexpected:
                    print(f"Warning: {len(unexpected)} unexpected keys in checkpoint. Example: {unexpected[:5]}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint from MDX23C_CKPT {ckpt_path}: {e}")
    
    model = model.to(device)
    model.eval()
    
    return model, config, device


def demix_audio(audio_path: str, 
                output_dir: str = None,
                checkpoint_path: str = None, 
                config_path: str = None,
                config = None,
                state_dict = None,
                device: str = None,
                use_tta: bool = False,
                extract_instrumental: bool = False,
                normalize: bool = False,
                sample_rate: int = 44100) -> dict:
    """
    High-level function to separate drums from an audio file.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems (if None, only returns arrays)
        checkpoint_path: Path to model checkpoint
        config_path: Path to config YAML file
        config: Config object directly (overrides config_path and env vars)
        state_dict: Model state dict directly (overrides checkpoint_path and env vars)
        device: Device to run on (auto-detected if None)
        use_tta: Whether to use test-time augmentation
        extract_instrumental: Whether to extract instrumental track
        normalize: Whether to normalize audio
        sample_rate: Target sample rate
        
    Returns:
        Dictionary mapping instrument names to audio arrays
    """
    import librosa
    import soundfile as sf
    
    # Load model - pass through all arguments
    model, config, device = load_model(
        checkpoint_path=checkpoint_path, 
        config_path=config_path, 
        config=config,
        state_dict=state_dict,
        device=device
    )
    
    # Override config with function parameters
    if normalize:
        config.inference.normalize = True
    config.audio.sample_rate = sample_rate
    
    # Load audio
    try:
        mix, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    except Exception as e:
        raise ValueError(f'Cannot read track: {audio_path}. Error: {str(e)}')
    
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if getattr(config.audio, 'num_channels', 2) == 2:
            mix = np.concatenate([mix, mix], axis=0)
    
    mix_orig = mix.copy()
    if getattr(config.inference, 'normalize', False):
        mix, norm_params = normalize_audio(mix)
    else:
        norm_params = None
    
    # Perform separation
    waveforms = demix(config, model, mix, device, model_type='mdx23c', pbar=True)
    
    # Apply TTA if requested
    if use_tta:
        waveforms = apply_tta(config, model, mix, waveforms, device, 'mdx23c')
    
    # Extract instrumental if requested
    instruments = prefer_target_instrument(config)[:]
    if extract_instrumental:
        instr = 'vocals' if 'vocals' in instruments else instruments[0]
        waveforms['instrumental'] = mix_orig - waveforms[instr]
        if 'instrumental' not in instruments:
            instruments.append('instrumental')
    
    # Denormalize if needed
    for instr in instruments:
        if getattr(config.inference, 'normalize', False) and norm_params is not None:
            waveforms[instr] = denormalize_audio(waveforms[instr], norm_params)
    
    # Save to files if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        for instr in instruments:
            estimates = waveforms[instr]
            output_path = os.path.join(output_dir, f"{file_name}_{instr}.wav")
            sf.write(output_path, estimates.T, sr, subtype='FLOAT')
            print(f"Saved: {output_path}")
    
    return waveforms


def demix_kit_from_mix(audio_path: str,
                      output_dir: str = None,
                      device: str = None,
                      use_tta: bool = False,
                      extract_instrumental: bool = False,
                      normalize: bool = False,
                      sample_rate: int = 44100,
                      force_download: bool = False) -> dict:
    """
    Convenience method to separate a full mix into drum kit components.
    
    Uses pre-trained model: model_mdx23c_ep_168_sdr_7.0207.ckpt
    Automatically downloads model files from Hugging Face if not cached.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems (if None, only returns arrays)
        device: Device to run on (auto-detected if None)
        use_tta: Whether to use test-time augmentation
        extract_instrumental: Whether to extract instrumental track
        normalize: Whether to normalize audio
        sample_rate: Target sample rate
        force_download: Whether to re-download model files even if cached
        
    Returns:
        Dictionary mapping instrument names to audio arrays
    """
    from .model_hub import load_model_from_hub
    
    print("🥁 Loading drum kit separation model...")
    model, config, device, _, _ = load_model_from_hub('kit_from_mix', device, force_download)
    
    # Use the loaded model directly with demix_audio
    return demix_audio(
        audio_path=audio_path,
        output_dir=output_dir,
        config=config,
        state_dict=model.state_dict(),
        device=device,
        use_tta=use_tta,
        extract_instrumental=extract_instrumental,
        normalize=normalize,
        sample_rate=sample_rate
    )


def demix_stems_from_kit(audio_path: str,
                        output_dir: str = None,
                        device: str = None,
                        use_tta: bool = False,
                        extract_instrumental: bool = False,
                        normalize: bool = False,
                        sample_rate: int = 44100,
                        force_download: bool = False) -> dict:
    """
    Convenience method to separate drum kit into individual stems.
    
    Uses pre-trained model: drumsep_5stems_mdx23c_jarredou.ckpt
    Automatically downloads model files from Hugging Face if not cached.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save separated stems (if None, only returns arrays)
        device: Device to run on (auto-detected if None)
        use_tta: Whether to use test-time augmentation
        extract_instrumental: Whether to extract instrumental track
        normalize: Whether to normalize audio
        sample_rate: Target sample rate
        force_download: Whether to re-download model files even if cached
        
    Returns:
        Dictionary mapping instrument names to audio arrays
    """
    from .model_hub import load_model_from_hub
    
    print("🥁 Loading drum stems separation model...")
    model, config, device, _, _ = load_model_from_hub('stems_from_kit', device, force_download)
    
    # Use the loaded model directly with demix_audio
    return demix_audio(
        audio_path=audio_path,
        output_dir=output_dir,
        config=config,
        state_dict=model.state_dict(),
        device=device,
        use_tta=use_tta,
        extract_instrumental=extract_instrumental,
        normalize=normalize,
        sample_rate=sample_rate
    )
