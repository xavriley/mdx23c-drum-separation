"""Command line interface for MDX23C drum separation."""

import argparse
import os
import glob
import time
import warnings
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm.auto import tqdm

from .inference import load_model, demix, apply_tta
from .config import prefer_target_instrument
from .audio_utils import normalize_audio, denormalize_audio, draw_spectrogram

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MDX23C Drum Separation")
    parser.add_argument("--input_folder", type=str, required=True, 
                       help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, required=True, 
                       help="path to store results")
    parser.add_argument("--checkpoint", type=str, default="",
                       help="path to model checkpoint (or set MDX23C_CKPT env var)")
    parser.add_argument("--config", type=str, default="", 
                       help="path to config YAML file (or set MDX23C_CONFIG env var)")
    parser.add_argument("--device", type=str, default="auto",
                       help="device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--draw_spectro", type=float, default=0.0,
                       help="length in seconds to draw spectrogram (0 to disable)")
    parser.add_argument("--extract_instrumental", action="store_true",
                       help="extract instrumental track")
    parser.add_argument("--use_tta", action="store_true",
                       help="use test-time augmentation")
    parser.add_argument("--flac_file", action="store_true",
                       help="save output as FLAC instead of WAV")
    parser.add_argument("--pcm_type", type=str, default="PCM_24",
                       choices=["PCM_16", "PCM_24"], help="PCM bit depth")
    parser.add_argument("--force_cpu", action="store_true",
                       help="force CPU usage")
    parser.add_argument("--disable_detailed_pbar", action="store_true",
                       help="disable detailed progress bars")
    parser.add_argument("--sample_rate", type=int, default=44100,
                       help="audio sample rate")
    parser.add_argument("--normalize", action="store_true",
                       help="normalize audio before processing")
    return parser.parse_args()


def initialize_device(args):
    """Initialize compute device."""
    if args.force_cpu or args.device == "cpu":
        return "cpu"
    elif args.device == "auto":
        if torch.cuda.is_available():
            print('CUDA is available, using GPU.')
            return 'cuda:0'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return "cpu"
    else:
        return args.device


def run_folder(model, config, args, device, verbose: bool = False):
    """Process all audio files in a folder."""
    start_time = time.time()
    model.eval()

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = args.sample_rate

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    detailed_pbar = not getattr(args, 'disable_detailed_pbar', False)

    for path in mixture_paths:
        print(f"Processing track: {path}")
        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {path}')
            print(f'Error message: {str(e)}')
            continue

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if getattr(config.audio, 'num_channels', 2) == 2:
                print('Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if args.normalize:
            mix, norm_params = normalize_audio(mix)
        else:
            norm_params = None

        waveforms_orig = demix(config, model, mix, device, model_type='mdx23c', pbar=detailed_pbar)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, 'mdx23c')

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(path))[0]
        output_dir = os.path.join(args.store_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if args.normalize and norm_params is not None:
                estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if args.flac_file else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = os.path.join(output_dir, f"{instr}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{instr}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def main():
    """Main CLI entry point."""
    
    args = parse_args()
    
    # Handle environment variables for checkpoint and config
    checkpoint_path = args.checkpoint or os.environ.get('MDX23C_CKPT', '').strip()
    config_path = args.config or os.environ.get('MDX23C_CONFIG', '').strip()
    
    # Initialize device
    device = initialize_device(args)
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True
    model_load_start_time = time.time()

    # Load model
    model, config, device = load_model(
        checkpoint_path=checkpoint_path if checkpoint_path else None,
        config_path=config_path if config_path else None, 
        device=device
    )
    
    # Override config with CLI args
    config.audio.sample_rate = args.sample_rate
    if args.normalize:
        config.inference.normalize = True

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))
    print("Instruments:", prefer_target_instrument(config))

    run_folder(model, config, args, device, verbose=True)


if __name__ == "__main__":
    main()
