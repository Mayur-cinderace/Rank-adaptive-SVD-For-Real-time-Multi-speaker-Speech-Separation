"""
backend/app/preprocessing.py

Shared preprocessing pipeline consumed by ALL three separation methods:
  - Beamforming (delay-and-sum / MVDR)
  - Neural (Conv-TasNet via asteroid)
  - Rank-Adaptive SVD (the novel contribution)

This module is the single source of truth for:
  - Sample rate (TARGET_SR = 16 000 Hz)
  - STFT parameters (window, hop, n_fft)
  - Normalisation strategy
  - Frame-matrix construction
  - iSTFT reconstruction

Any change here propagates to all three methods automatically, ensuring
a fair apples-to-apples comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.signal as ss

# ─────────────────────────────────────────────────────────────────────────────
# Global constants — DO NOT change per-method; change here only
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR: int = 16_000          # Hz — all audio resampled to this
N_FFT: int = 512                 # STFT FFT size
HOP_LENGTH: int = 128            # STFT hop (25 % overlap of N_FFT)
WIN_LENGTH: int = 512            # STFT analysis window length
WINDOW: str = "hann"             # Window type (scipy name)
FRAME_DURATION_MS: float = 256.0 # Time-domain frame size for SVD (ms)
FRAME_HOP_MS: float = 64.0       # Time-domain frame hop for SVD (ms)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreparedSignal:
    """
    Output of the shared preprocessing stage.
    Carries everything the separation methods need.
    """
    # Raw multichannel audio: shape (M, T) float32
    waveform: np.ndarray

    # Per-channel STFT: shape (M, n_bins, n_frames) complex64
    stft_matrix: np.ndarray

    # Scalar metadata
    sr: int
    n_channels: int
    n_samples: int
    duration_sec: float
    n_fft: int
    hop_length: int
    win_length: int

    # Reference mono mix (channel mean) for metric computation
    mono_mix: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))


@dataclass
class SeparationResult:
    """
    Common output format for every separation method.
    Makes metric computation uniform.
    """
    method: str                          # "beamforming" | "svd" | "neural"
    separated_channels: list[np.ndarray] # list of 1-D float32 arrays (one per estimated source)
    sr: int
    metadata: dict = field(default_factory=dict)

    # Filled in by evaluate()
    stoi_scores: list[float] = field(default_factory=list)
    pesq_scores: list[float] = field(default_factory=list)
    sdr: Optional[float] = None
    sir: Optional[float] = None
    sar: Optional[float] = None
    rtf: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_waveform(x: np.ndarray, headroom_db: float = 1.0) -> np.ndarray:
    """
    Peak-normalize to -headroom_db dBFS.
    Applied identically to every method's input.
    """
    x = np.asarray(x, dtype=np.float32)
    peak = np.max(np.abs(x))
    if peak > 1e-8:
        target = 10 ** (-headroom_db / 20.0)
        x = x * (target / peak)
    return x


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
) -> np.ndarray:
    """
    Compute STFT for a single-channel or multi-channel waveform.

    Parameters
    ----------
    waveform : (T,) or (M, T) float32
    Returns
    -------
    stft : (n_bins, n_frames) complex64  — if input is 1-D
           (M, n_bins, n_frames) complex64 — if input is 2-D
    """
    win = ss.get_window(window, win_length, fftbins=True)
    win = np.pad(win, (0, n_fft - win_length)) if n_fft > win_length else win

    def _stft_1d(x: np.ndarray) -> np.ndarray:
        _, _, Z = ss.stft(
            x.astype(np.float32),
            fs=TARGET_SR,
            window=win,
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        return Z.astype(np.complex64)

    if waveform.ndim == 1:
        return _stft_1d(waveform)

    return np.stack([_stft_1d(waveform[m]) for m in range(waveform.shape[0])], axis=0)


def compute_istft(
    stft_matrix: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    window: str = WINDOW,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Inverse STFT.

    Parameters
    ----------
    stft_matrix : (n_bins, n_frames) complex64
    Returns
    -------
    waveform : (T,) float32
    """
    win = ss.get_window(window, win_length, fftbins=True)
    win = np.pad(win, (0, n_fft - win_length)) if n_fft > win_length else win

    _, x = ss.istft(
        stft_matrix.astype(np.complex64),
        fs=TARGET_SR,
        window=win,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=True,
    )
    x = x.astype(np.float32)
    if length is not None:
        x = x[:length] if len(x) >= length else np.pad(x, (0, length - len(x)))
    return x


def build_time_domain_frames(
    signal: np.ndarray,
    frame_ms: float = FRAME_DURATION_MS,
    hop_ms: float = FRAME_HOP_MS,
    sr: int = TARGET_SR,
) -> tuple[np.ndarray, int, int]:
    """
    Segment a 1-D signal into overlapping frames for time-domain SVD.

    Returns
    -------
    frames    : (n_frames, frame_len) float32
    frame_len : int
    hop_len   : int
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    n = len(signal)
    n_frames = max(1, 1 + (n - frame_len) // hop_len)

    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        chunk = signal[start:min(end, n)]
        frames[i, : len(chunk)] = chunk
    return frames, frame_len, hop_len


def overlap_add(
    frames: np.ndarray,
    frame_len: int,
    hop_len: int,
    total_len: int,
) -> np.ndarray:
    """
    Overlap-add reconstruction from time-domain frames.

    Parameters
    ----------
    frames : (n_frames, frame_len)
    Returns
    -------
    signal : (total_len,) float32
    """
    out = np.zeros(total_len, dtype=np.float32)
    norm = np.zeros(total_len, dtype=np.float32)
    win = np.hanning(frame_len).astype(np.float32)

    for i, frame in enumerate(frames):
        start = i * hop_len
        end = min(start + frame_len, total_len)
        out[start:end] += (frame * win)[: end - start]
        norm[start:end] += win[: end - start]

    nonzero = norm > 1e-8
    out[nonzero] /= norm[nonzero]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def prepare(
    X_raw: np.ndarray,
    sr: int,
    *,
    do_normalize: bool = True,
) -> PreparedSignal:
    """
    Run the shared preprocessing pipeline.

    Parameters
    ----------
    X_raw : (M, T) or (T,) float32 — raw multichannel audio
    sr    : original sample rate (already resampled to TARGET_SR by audio_service)
    do_normalize : whether to peak-normalize (True for all fair comparisons)

    Returns
    -------
    PreparedSignal with both waveform and STFT representations ready for
    beamforming, SVD, and neural methods.
    """
    # Ensure 2-D: (M, T)
    if X_raw.ndim == 1:
        X_raw = X_raw[np.newaxis, :]
    X = np.asarray(X_raw, dtype=np.float32)

    # Peak-normalize across all channels jointly
    if do_normalize:
        X = normalize_waveform(X)

    # Mono mix (used as reference signal for blind evaluation)
    mono = np.mean(X, axis=0).astype(np.float32)

    # STFT for all channels
    stft = compute_stft(
        X,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
    )  # (M, n_bins, n_frames)

    n_channels, n_samples = X.shape
    duration = n_samples / sr

    return PreparedSignal(
        waveform=X,
        stft_matrix=stft,
        sr=sr,
        n_channels=n_channels,
        n_samples=n_samples,
        duration_sec=duration,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        mono_mix=mono,
    )