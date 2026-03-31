"""
backend/app/separation_svd.py

Rank-Adaptive SVD source separation — the novel contribution.

Implements the methodology from the research design document:
  - Time-domain framing: X ∈ R^(M × T) per frame
  - SVD: X = U Σ V^T
  - Adaptive rank selection via cumulative energy threshold τ
  - Overlap-add reconstruction
  - Optional Wiener post-filter using the singular value tail as noise estimate

All STFT parameters and frame sizes come from preprocessing.py to ensure
the comparison with beamforming and neural methods is fair.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .preprocessing import (
    PreparedSignal,
    SeparationResult,
    build_time_domain_frames,
    compute_istft,
    compute_stft,
    overlap_add,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDConfig:
    """All tuneable hyperparameters for the rank-adaptive SVD method."""

    # Energy threshold τ — tuned on LibriMix dev set (sweep 0.80–0.95)
    tau: float = 0.90

    # If True, use frequency-domain STFT frames (better spectral resolution)
    # If False, use time-domain frames (lower latency, as in architecture diagram)
    frequency_domain: bool = True

    # Wiener post-filter using singular value tail as noise PSD estimate
    use_wiener: bool = True

    # Maximum number of sources to separate (safety cap)
    max_sources: int = 4

    # Minimum rank (always reconstruct at least 1 component)
    min_rank: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Core rank selection (the novel rule — Equation in §6 of research doc)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_rank(singular_values: np.ndarray, tau: float = 0.90) -> int:
    """
    Returns r*, the minimum rank capturing fraction tau of total energy.

    ε_k = σ_k² / Σ_i σ_i²
    r* = min { r : Σ_{k=1}^{r} ε_k ≥ τ }

    Parameters
    ----------
    singular_values : (M,) float — sorted descending (guaranteed by np.linalg.svd)
    tau             : energy threshold in (0, 1)

    Returns
    -------
    r_star : int — adaptive rank for this frame
    """
    energy = singular_values ** 2
    total = energy.sum()
    if total < 1e-12:
        return 1
    cumulative = np.cumsum(energy) / total
    r_star = int(np.searchsorted(cumulative, tau)) + 1
    return min(r_star, len(singular_values))


def wiener_gain(
    singular_values: np.ndarray,
    r_star: int,
) -> np.ndarray:
    """
    Wiener filter weights derived from singular value spectrum.

    Signal PSD ≈ σ_k² for k ≤ r*
    Noise PSD  ≈ mean(σ_k²) for k > r*  (tail estimate)

    W_k = σ_k² / (σ_k² + σ_noise²)

    Returns
    -------
    gains : (r_star,) float in [0, 1]
    """
    energy = singular_values ** 2
    tail = energy[r_star:]
    noise_psd = float(np.mean(tail)) if len(tail) > 0 else 0.0

    signal_psd = energy[:r_star]
    gains = signal_psd / (signal_psd + noise_psd + 1e-10)
    return gains.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain SVD (preferred: better spectral separation)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_frequency_domain(
    prep: PreparedSignal,
    cfg: SVDConfig,
) -> list[np.ndarray]:
    """
    Process each STFT frequency bin independently with SVD.

    X_f ∈ C^(M × N_frames)  for each frequency bin f
    → SVD → adaptive rank selection → rank-r* reconstruction → iSTFT

    Returns
    -------
    sources : list of (T,) float32 — one array per estimated source
    """
    M, n_bins, n_frames = prep.stft_matrix.shape  # (M, F, N)

    # We separate up to max_sources — for each source, hold its STFT
    n_sources = min(cfg.max_sources, M)
    source_stfts = [
        np.zeros((n_bins, n_frames), dtype=np.complex64)
        for _ in range(n_sources)
    ]

    # Track per-frame r* for diagnostics
    r_stars: list[int] = []

    for f in range(n_bins):
        # X_f: (M, N_frames)
        Xf = prep.stft_matrix[:, f, :]  # complex64

        # SVD — economy form
        try:
            U, s, Vh = np.linalg.svd(Xf, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        r_star = max(
            cfg.min_rank,
            min(adaptive_rank(s, cfg.tau), n_sources),
        )
        r_stars.append(r_star)

        gains = wiener_gain(s, r_star) if cfg.use_wiener else np.ones(r_star)

        # Reconstruct each source component
        for k in range(min(r_star, n_sources)):
            # Rank-1 outer product for component k, scaled by Wiener gain
            comp = gains[k] * s[k] * np.outer(U[:, k], Vh[k, :])
            # Project onto channel 0 (target speaker channel)
            source_stfts[k][f, :] += comp[0, :]

    logger.debug(
        "SVD freq-domain: mean r* = %.2f over %d bins",
        float(np.mean(r_stars)) if r_stars else 0.0,
        n_bins,
    )

    # iSTFT for each source
    sources: list[np.ndarray] = []
    for k in range(n_sources):
        wav = compute_istft(
            source_stfts[k],
            n_fft=prep.n_fft,
            hop_length=prep.hop_length,
            win_length=prep.win_length,
            length=prep.n_samples,
        )
        sources.append(wav)

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain SVD (lower latency variant, matches architecture diagram §7)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_time_domain(
    prep: PreparedSignal,
    cfg: SVDConfig,
) -> list[np.ndarray]:
    """
    Form X ∈ R^(M × T_frame) per time frame and apply rank-adaptive SVD.

    Architecture matches §7 of the research doc:
      X  →  SVD  →  Rank Selector  →  Rank-r* Reconstruction  →  OLA

    Returns
    -------
    sources : list of (T,) float32
    """
    M, T = prep.waveform.shape
    n_sources = min(cfg.max_sources, M)

    # Use channel 0 as reference for framing parameters
    frames_ref, frame_len, hop_len = build_time_domain_frames(prep.waveform[0])
    n_frames = len(frames_ref)

    # Accumulated per-source frames for overlap-add
    source_frames = [
        np.zeros((n_frames, frame_len), dtype=np.float32)
        for _ in range(n_sources)
    ]

    r_stars: list[int] = []

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len

        # Build observation matrix X ∈ R^(M × frame_len)
        X = np.zeros((M, frame_len), dtype=np.float32)
        for m in range(M):
            chunk = prep.waveform[m, start:min(end, T)]
            X[m, : len(chunk)] = chunk

        try:
            U, s, Vh = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        r_star = max(
            cfg.min_rank,
            min(adaptive_rank(s, cfg.tau), n_sources),
        )
        r_stars.append(r_star)

        gains = wiener_gain(s, r_star) if cfg.use_wiener else np.ones(r_star)

        # Reconstruct each source from its singular triplet
        for k in range(min(r_star, n_sources)):
            # Rank-1 approximation: component k in row-space of X
            # Row 0 of the reconstruction = target channel contribution
            comp_row0 = gains[k] * s[k] * U[0, k] * Vh[k, :]
            source_frames[k][i] = comp_row0.astype(np.float32)

    logger.debug(
        "SVD time-domain: mean r* = %.2f over %d frames",
        float(np.mean(r_stars)) if r_stars else 0.0,
        n_frames,
    )

    # Overlap-add reconstruction
    sources: list[np.ndarray] = []
    for k in range(n_sources):
        wav = overlap_add(source_frames[k], frame_len, hop_len, T)
        sources.append(wav)

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_svd_separation(
    prep: PreparedSignal,
    cfg: SVDConfig | None = None,
) -> SeparationResult:
    """
    Run rank-adaptive SVD separation on a PreparedSignal.

    Parameters
    ----------
    prep : output of preprocessing.prepare()
    cfg  : SVDConfig — uses defaults (τ=0.90, freq-domain, Wiener on) if None

    Returns
    -------
    SeparationResult with separated_channels list
    """
    if cfg is None:
        cfg = SVDConfig()

    t0 = time.perf_counter()

    if prep.n_channels < 2:
        logger.warning("SVD: only 1 channel — separation is degenerate (returning as-is).")
        return SeparationResult(
            method="svd",
            separated_channels=[prep.waveform[0].copy()],
            sr=prep.sr,
            metadata={"warning": "single_channel", "tau": cfg.tau},
        )

    if cfg.frequency_domain:
        sources = _svd_frequency_domain(prep, cfg)
    else:
        sources = _svd_time_domain(prep, cfg)

    elapsed = time.perf_counter() - t0
    rtf = elapsed / prep.duration_sec if prep.duration_sec > 0 else None

    # Compute per-frame r* diagnostics (stored in metadata for the frontend)
    # Re-derive quickly from the mono mix for logging
    _, s_diag, _ = np.linalg.svd(prep.waveform, full_matrices=False)
    r_global = adaptive_rank(s_diag, cfg.tau)

    return SeparationResult(
        method="svd",
        separated_channels=sources,
        sr=prep.sr,
        rtf=rtf,
        metadata={
            "tau": cfg.tau,
            "frequency_domain": cfg.frequency_domain,
            "wiener": cfg.use_wiener,
            "global_rank_estimate": r_global,
            "n_sources_returned": len(sources),
            "processing_time_sec": round(elapsed, 4),
        },
    )