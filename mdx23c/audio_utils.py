"""Audio processing utilities."""

import numpy as np
import os


def normalize_audio(audio: np.ndarray):
    """Normalize audio array."""
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / (std if std != 0 else 1.0), {"mean": mean, "std": std if std != 0 else 1.0}


def denormalize_audio(audio: np.ndarray, norm_params):
    """Denormalize audio array using stored parameters."""
    return audio * norm_params["std"] + norm_params["mean"]


def draw_spectrogram(waveform: np.ndarray, sample_rate: int, length: float, output_file: str) -> None:
    """Draw and save spectrogram of waveform."""
    import matplotlib.pyplot as plt
    import librosa.display
    import librosa

    x = waveform[:int(length * sample_rate), :]
    X = librosa.stft(x.mean(axis=-1))
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        Xdb, cmap='plasma', sr=sample_rate, x_axis='time', y_axis='linear', ax=ax
    )
    ax.set(title='File: ' + os.path.basename(output_file))
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    if output_file is not None:
        plt.savefig(output_file)
