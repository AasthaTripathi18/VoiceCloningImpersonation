"""
preprocess.py

Audio loading / cleaning utilities shared by training, evaluation, and
live inference. Keeping this centralized matters: any mismatch between
how training audio and live-call audio are preprocessed will quietly
hurt accuracy.

(Unchanged from the training notebook -- deployment MUST use the exact
same preprocessing as training, or accuracy will silently degrade.)
"""

import torch
import torchaudio
import torchaudio.functional as AF


TARGET_SR = 16000  # Wav2Vec2 expects 16kHz mono input


def load_audio(filepath: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Load an audio file, downmix to mono, resample to target_sr."""
    waveform, sr = torchaudio.load(filepath)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    return waveform.squeeze(0)  # (num_samples,)


def trim_silence(waveform: torch.Tensor, top_db: float = 30.0) -> torch.Tensor:
    """Trim leading/trailing silence. Falls back gracefully on pure silence."""
    import librosa

    audio_np = waveform.numpy()
    trimmed, _ = librosa.effects.trim(audio_np, top_db=top_db)
    if len(trimmed) == 0:
        return waveform
    return torch.from_numpy(trimmed)


def normalize_volume(waveform: torch.Tensor) -> torch.Tensor:
    """Peak-normalize to avoid loudness being a spurious signal."""
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


def pad_or_truncate(waveform: torch.Tensor, max_duration: float = 4.0, sr: int = TARGET_SR) -> torch.Tensor:
    """Fix all clips to the same length for batching."""
    max_samples = int(max_duration * sr)
    if waveform.shape[0] > max_samples:
        return waveform[:max_samples]
    pad_amount = max_samples - waveform.shape[0]
    return torch.nn.functional.pad(waveform, (0, pad_amount))


def simulate_phone_channel(waveform: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    """Approximate phone-codec narrowband audio. Training-time augmentation only."""
    waveform = AF.highpass_biquad(waveform, sr, cutoff_freq=300)
    waveform = AF.lowpass_biquad(waveform, sr, cutoff_freq=3400)
    noise = torch.randn_like(waveform) * 0.005
    waveform = waveform + noise
    return waveform


def preprocess_pipeline(
    filepath: str,
    target_sr: int = TARGET_SR,
    max_duration: float = 4.0,
    trim: bool = True,
    augment_phone_channel: bool = False,
) -> torch.Tensor:
    waveform = load_audio(filepath, target_sr)
    if trim:
        waveform = trim_silence(waveform)
    waveform = normalize_volume(waveform)
    if augment_phone_channel:
        waveform = simulate_phone_channel(waveform, target_sr)
    waveform = pad_or_truncate(waveform, max_duration, target_sr)
    return waveform
