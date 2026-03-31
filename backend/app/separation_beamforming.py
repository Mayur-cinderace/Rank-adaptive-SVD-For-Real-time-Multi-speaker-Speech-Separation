"""
backend/app/separation_beamforming.py

Classical beamforming separation — two variants:
  1. Delay-and-Sum (DAS)  — simple, steers by integer sample shifts
  2. MVDR (Capon)         — data-adaptive, suppresses interference

Both consume PreparedSignal from preprocessing.py and return a
SeparationResult with the same schema as the SVD and neural methods.

Key design choices for fair comparison:
  - No oracle DOA is assumed. We estimate delays from the observed
    cross-correlation of the STFT across channels (GCC-PHAT).
  - The same N_FFT / HOP_LENGTH from preprocessing.py are used.
  - MVDR uses the same STFT frames as the SVD method.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .preprocessing import (
    PreparedSignal,
    SeparationResult,
    compute_istft,
)

logger = logging.getLogger(__name__)

SPEED_OF_SOUND: float = 343.0      # m/s (room temperature)
MIC_SPACING: float = 0.05          # 5 cm default inter-mic spacing (m)
DIAGONAL_LOADING: float = 1e-3     # MVDR diagonal loading factor


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeamformingConfig:
    variant: str = "mvdr"          # "das" | "mvdr"
    mic_spacing_m: float = MIC_SPACING
    # If None, delays are estimated from GCC-PHAT
    # If provided (list of ints), used directly (oracle / synthetic mode)
    known_delays_samples: list[int] | None = None
    diagonal_loading: float = DIAGONAL_LOADING


# ─────────────────────────────────────────────────────────────────────────────
# GCC-PHAT delay estimation (no oracle DOA needed)
# ─────────────────────────────────────────────────────────────────────────────

def gcc_phat_delay(ref: np.ndarray, sig: np.ndarray, sr: int, max_delay: int | None = None) -> int:
    """
    Estimate the integer-sample delay between ref and sig using GCC-PHAT.

    Returns
    -------
    delay : int — positive means sig leads ref (advance sig by `delay` samples)
    """
    n = len(ref) + len(sig) - 1
    n_fft = int(2 ** np.ceil(np.log2(n)))

    R = np.fft.rfft(ref, n=n_fft) * np.conj(np.fft.rfft(sig, n=n_fft))
    denom = np.abs(R) + 1e-10
    gcc = np.fft.irfft(R / denom, n=n_fft)

    if max_delay is None:
        max_delay = sr // 100  # cap at 10 ms

    # Search window: ±max_delay samples around zero-lag
    search = np.concatenate([gcc[-max_delay:], gcc[:max_delay + 1]])
    peak = np.argmax(np.abs(search))
    delay = peak - max_delay   # signed delay
    return int(delay)


def estimate_delays(prep: PreparedSignal, mic_spacing_m: float) -> list[int]:
    """
    Estimate inter-channel delays using GCC-PHAT between each channel
    and the reference channel (channel 0).

    Returns
    -------
    delays : list[int] of length M — delays[0] = 0 (reference)
    """
    M = prep.n_channels
    ref = prep.waveform[0]
    max_delay = int(mic_spacing_m * (M - 1) * prep.sr / SPEED_OF_SOUND) + 10
    delays = [0]
    for m in range(1, M):
        d = gcc_phat_delay(ref, prep.waveform[m], prep.sr, max_delay=max_delay)
        delays.append(d)
    logger.debug("GCC-PHAT estimated delays (samples): %s", delays)
    return delays


# ─────────────────────────────────────────────────────────────────────────────
# Delay-and-Sum beamforming
# ─────────────────────────────────────────────────────────────────────────────

def _delay_and_sum(prep: PreparedSignal, delays: list[int]) -> np.ndarray:
    """
    Align channels by their estimated delays and average.

    Returns
    -------
    output : (T,) float32
    """
    M, T = prep.waveform.shape
    aligned = np.zeros((M, T), dtype=np.float32)

    for m, d in enumerate(delays):
        sig = prep.waveform[m]
        if d == 0:
            aligned[m] = sig
        elif d > 0:
            # sig leads ref — shift right (delay sig)
            aligned[m, d:] = sig[:-d] if d < T else 0
        else:
            # sig lags ref — advance sig
            d_abs = -d
            aligned[m, :T - d_abs] = sig[d_abs:]

    return np.mean(aligned, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MVDR beamforming (Capon / minimum variance distortionless response)
# ─────────────────────────────────────────────────────────────────────────────

def _build_steering_vector(delays: list[int], n_bins: int, sr: int, n_fft: int) -> np.ndarray:
    """
    Construct the steering vector a(f) ∈ C^M for each frequency bin.

    a_m(f) = exp(-j 2π f τ_m)

    where τ_m = delays[m] / sr (seconds)

    Returns
    -------
    A : (n_bins, M) complex64
    """
    M = len(delays)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)   # (n_bins,)
    taus = np.array(delays, dtype=np.float32) / sr  # (M,)
    # A[f, m] = exp(-j 2π freq[f] * tau[m])
    A = np.exp(-1j * 2 * np.pi * freqs[:, np.newaxis] * taus[np.newaxis, :])
    return A.astype(np.complex64)


def _mvdr(
    prep: PreparedSignal,
    delays: list[int],
    diagonal_loading: float,
) -> np.ndarray:
    """
    MVDR beamformer in the STFT domain.

    For each frequency bin f:
      R_f = (1/N) Σ_t X_f(:,t) X_f(:,t)^H  — spatial covariance
      w_f = R_f^{-1} a_f / (a_f^H R_f^{-1} a_f)  — MVDR weights
      Y_f(t) = w_f^H X_f(:, t)                     — beamformed output

    Returns
    -------
    output : (T,) float32
    """
    M, n_bins, n_frames = prep.stft_matrix.shape   # (M, F, N)

    A = _build_steering_vector(delays, n_bins, prep.sr, prep.n_fft)  # (F, M)

    out_stft = np.zeros((n_bins, n_frames), dtype=np.complex64)

    for f in range(n_bins):
        Xf = prep.stft_matrix[:, f, :]   # (M, N) complex

        # Spatial covariance: (M, M)
        R = (Xf @ Xf.conj().T) / n_frames
        # Diagonal loading for numerical stability
        R += diagonal_loading * np.eye(M, dtype=np.complex64)

        a = A[f]   # (M,) complex steering vector

        try:
            Rinv_a = np.linalg.solve(R, a)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            Rinv_a = np.linalg.pinv(R) @ a

        denom = (a.conj() @ Rinv_a).real + 1e-10
        w = Rinv_a / denom   # (M,) MVDR weight vector

        # Apply weights: Y_f(t) = w^H X_f(:, t)
        out_stft[f, :] = w.conj() @ Xf   # (N,)

    return compute_istft(
        out_stft,
        n_fft=prep.n_fft,
        hop_length=prep.hop_length,
        win_length=prep.win_length,
        length=prep.n_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_beamforming(
    prep: PreparedSignal,
    cfg: BeamformingConfig | None = None,
) -> SeparationResult:
    """
    Run beamforming on a PreparedSignal.

    Parameters
    ----------
    prep : output of preprocessing.prepare()
    cfg  : BeamformingConfig (defaults: MVDR, GCC-PHAT delay estimation)

    Returns
    -------
    SeparationResult with a single separated channel (the beamformed output)
    """
    if cfg is None:
        cfg = BeamformingConfig()

    t0 = time.perf_counter()

    if prep.n_channels < 2:
        logger.warning("Beamforming: only 1 channel — returning channel 0 as-is.")
        return SeparationResult(
            method="beamforming",
            separated_channels=[prep.waveform[0].copy()],
            sr=prep.sr,
            metadata={"warning": "single_channel", "variant": cfg.variant},
        )

    # Delay estimation
    if cfg.known_delays_samples is not None:
        delays = cfg.known_delays_samples
        delay_source = "oracle"
    else:
        delays = estimate_delays(prep, cfg.mic_spacing_m)
        delay_source = "gcc_phat"

    # Beamform
    if cfg.variant == "das":
        output = _delay_and_sum(prep, delays)
    else:
        output = _mvdr(prep, delays, cfg.diagonal_loading)

    elapsed = time.perf_counter() - t0
    rtf = elapsed / prep.duration_sec if prep.duration_sec > 0 else None

    return SeparationResult(
        method="beamforming",
        separated_channels=[output],
        sr=prep.sr,
        rtf=rtf,
        metadata={
            "variant": cfg.variant,
            "delays_samples": delays,
            "delay_source": delay_source,
            "mic_spacing_m": cfg.mic_spacing_m,
            "processing_time_sec": round(elapsed, 4),
        },
    )