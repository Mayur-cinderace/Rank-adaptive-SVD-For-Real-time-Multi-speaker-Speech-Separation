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
from dataclasses import dataclass

import numpy as np

from .preprocessing import (
    PreparedSignal,
    SeparationResult,
    build_time_domain_frames,
    compute_istft,
    overlap_add,
)

logger = logging.getLogger(__name__)

SVD_EPS: float = 1e-6
LOW_ENERGY_EPS: float = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SVDConfig:
    """All tuneable hyperparameters for the rank-adaptive SVD method."""

    # Energy threshold τ — tuned on LibriMix dev set (sweep 0.80–0.95)
    tau: float = 0.85

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

def robust_rank(
    singular_values: np.ndarray,
    tau: float = 0.90,
    min_ratio: float = 0.1,
    max_sources: int = 3,
) -> int:
    """Hybrid rank rule: cumulative energy + relative strength with stability cap."""
    energy = singular_values ** 2
    total = float(energy.sum())

    if total < LOW_ENERGY_EPS:
        return 1

    cumulative = np.cumsum(energy) / total
    r_energy = int(np.searchsorted(cumulative, tau)) + 1

    ratios = singular_values / (float(singular_values[0]) + LOW_ENERGY_EPS)
    r_ratio = int(np.sum(ratios > min_ratio))

    r_star = min(r_energy, r_ratio, max_sources)
    return max(1, r_star)


def _smooth_rank_trace(ranks: list[int], window: int = 5) -> list[int]:
    """Moving-average smoothing to stabilize per-frame rank estimates."""
    if not ranks:
        return []
    arr = np.asarray(ranks, dtype=np.float32)
    if len(arr) < window:
        return [int(max(1, round(v))) for v in arr]

    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(arr, kernel, mode="same")
    return [int(max(1, round(v))) for v in smoothed]


def _postprocess_sources(sources: list[np.ndarray], keep_top_n: int = 2) -> tuple[list[np.ndarray], list[float]]:
    """Remove DC, normalize, then keep strongest perceptually distinct sources."""
    if not sources:
        return [], []

    dc_removed: list[np.ndarray] = []
    energies: list[float] = []
    for src in sources:
        x = np.asarray(src, dtype=np.float32).copy()
        x -= float(np.mean(x))
        e = float(np.mean(x ** 2))
        dc_removed.append(x)
        energies.append(e)

    if not energies:
        return [], []

    max_energy = max(energies)
    keep = [i for i, e in enumerate(energies) if e >= 0.1 * max_energy]
    if not keep:
        keep = [int(np.argmax(energies))]

    keep = sorted(keep, key=lambda i: energies[i], reverse=True)[: max(1, keep_top_n)]
    kept_sources = [dc_removed[i] for i in keep]
    kept_energies = [energies[i] for i in keep]

    # Final per-source normalization for stable listening level.
    for i, src in enumerate(kept_sources):
        src /= float(np.max(np.abs(src)) + 1e-8)
        kept_sources[i] = src.astype(np.float32)

    denom = float(sum(kept_energies)) + 1e-10
    energy_distribution = [float(e / denom) for e in kept_energies]
    return kept_sources, energy_distribution


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
) -> tuple[list[np.ndarray], dict]:
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
    n_sources = min(3, M, cfg.max_sources)
    source_stfts = [
        np.zeros((n_bins, n_frames), dtype=np.complex64)
        for _ in range(n_sources)
    ]

    raw_ranks: list[int] = []
    svd_bins: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    singular_bank: list[np.ndarray] = []

    for f in range(n_bins):
        Xf = prep.stft_matrix[:, f, :]  # complex64
        frame_energy = float(np.mean(np.abs(Xf) ** 2))
        if frame_energy < LOW_ENERGY_EPS:
            raw_ranks.append(1)
            svd_bins.append(None)
            continue

        Xf_reg = Xf + SVD_EPS
        try:
            U, s, Vh = np.linalg.svd(Xf_reg, full_matrices=False)
        except np.linalg.LinAlgError:
            raw_ranks.append(1)
            svd_bins.append(None)
            continue

        r_star = robust_rank(s, tau=cfg.tau, min_ratio=0.1, max_sources=min(3, n_sources))
        r_star = max(cfg.min_rank, min(r_star, n_sources))
        raw_ranks.append(r_star)
        svd_bins.append((U, s, Vh))
        singular_bank.append(s)

    smooth_ranks = _smooth_rank_trace(raw_ranks, window=5)

    for f in range(n_bins):
        svd_parts = svd_bins[f]
        if svd_parts is None:
            continue
        U, s, Vh = svd_parts
        r_star = max(cfg.min_rank, min(smooth_ranks[f], n_sources, len(s)))
        gains = wiener_gain(s, r_star) if cfg.use_wiener else np.ones(r_star, dtype=np.float32)

        for k in range(min(r_star, n_sources)):
            uk = U[:, k]
            vk = Vh[k, :]
            weights = np.abs(uk)
            wsum = float(np.sum(weights))
            if wsum <= LOW_ENERGY_EPS:
                continue
            weights = weights / wsum
            mix_coeff = np.sum(weights * uk)
            component = gains[k] * s[k] * mix_coeff * vk
            source_stfts[k][f, :] += component.astype(np.complex64)

    logger.debug(
        "SVD freq-domain: mean r* = %.2f over %d bins",
        float(np.mean(smooth_ranks)) if smooth_ranks else 0.0,
        n_bins,
    )

    # iSTFT for each source
    sources_raw: list[np.ndarray] = []
    for k in range(n_sources):
        wav = compute_istft(
            source_stfts[k],
            n_fft=prep.n_fft,
            hop_length=prep.hop_length,
            win_length=prep.win_length,
            length=prep.n_samples,
        )
        sources_raw.append(wav)

    sources, energy_distribution = _postprocess_sources(sources_raw, keep_top_n=2)

    avg_rank = int(np.round(np.mean(smooth_ranks))) if smooth_ranks else 1
    if singular_bank:
        first10 = np.zeros(10, dtype=np.float64)
        for s in singular_bank:
            take = min(10, len(s))
            first10[:take] += s[:take]
        first10 /= max(1, len(singular_bank))
    else:
        first10 = np.zeros(10, dtype=np.float64)
    if first10[0] > 0:
        first10 = first10 / first10[0]

    diagnostics = {
        "rank_trace": [int(v) for v in smooth_ranks],
        "rank_trace_raw": [int(v) for v in raw_ranks],
        "rank_per_frame": [int(v) for v in smooth_ranks],
        "rank_trace_domain": "frequency_bins",
        "first_10_singular_values": [float(v) for v in first10.tolist()],
        "singular_value_spectrum": [float(v) for v in first10.tolist()],
        "energy_distribution": energy_distribution,
        "estimated_sources": avg_rank,
    }
    return sources, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain SVD (lower latency variant, matches architecture diagram §7)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_time_domain(
    prep: PreparedSignal,
    cfg: SVDConfig,
) -> tuple[list[np.ndarray], dict]:
    """
    Form X ∈ R^(M × T_frame) per time frame and apply rank-adaptive SVD.

    Architecture matches §7 of the research doc:
      X  →  SVD  →  Rank Selector  →  Rank-r* Reconstruction  →  OLA

    Returns
    -------
    sources : list of (T,) float32
    """
    M, T = prep.waveform.shape
    n_sources = min(3, M, cfg.max_sources)

    # Use channel 0 as reference for framing parameters
    frames_ref, frame_len, hop_len = build_time_domain_frames(prep.waveform[0])
    n_frames = len(frames_ref)

    # Accumulated per-source frames for overlap-add
    source_frames = [
        np.zeros((n_frames, frame_len), dtype=np.float32)
        for _ in range(n_sources)
    ]

    raw_ranks: list[int] = []
    svd_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    singular_bank: list[np.ndarray] = []

    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len

        # Build observation matrix X ∈ R^(M × frame_len)
        X = np.zeros((M, frame_len), dtype=np.float32)
        for m in range(M):
            chunk = prep.waveform[m, start:min(end, T)]
            X[m, : len(chunk)] = chunk

        frame_energy = float(np.mean(X ** 2))
        if frame_energy < LOW_ENERGY_EPS:
            raw_ranks.append(1)
            svd_frames.append(None)
            continue

        X_reg = X + SVD_EPS

        try:
            U, s, Vh = np.linalg.svd(X_reg, full_matrices=False)
        except np.linalg.LinAlgError:
            raw_ranks.append(1)
            svd_frames.append(None)
            continue

        r_star = robust_rank(s, tau=cfg.tau, min_ratio=0.1, max_sources=min(3, n_sources))
        r_star = max(cfg.min_rank, min(r_star, n_sources))
        raw_ranks.append(r_star)
        svd_frames.append((U, s, Vh))
        singular_bank.append(s)

    smooth_ranks = _smooth_rank_trace(raw_ranks, window=5)

    for i in range(n_frames):
        svd_parts = svd_frames[i]
        if svd_parts is None:
            continue
        U, s, Vh = svd_parts
        r_star = max(cfg.min_rank, min(smooth_ranks[i], n_sources, len(s)))
        gains = wiener_gain(s, r_star) if cfg.use_wiener else np.ones(r_star, dtype=np.float32)

        for k in range(min(r_star, n_sources)):
            uk = U[:, k]
            vk = Vh[k, :]
            weights = np.abs(uk)
            wsum = float(np.sum(weights))
            if wsum <= LOW_ENERGY_EPS:
                continue
            weights = weights / wsum
            mix_coeff = np.sum(weights * uk)
            component = gains[k] * s[k] * mix_coeff * vk
            source_frames[k][i] = np.asarray(np.real(component), dtype=np.float32)

    logger.debug(
        "SVD time-domain: mean r* = %.2f over %d frames",
        float(np.mean(smooth_ranks)) if smooth_ranks else 0.0,
        n_frames,
    )

    # Overlap-add reconstruction
    sources_raw: list[np.ndarray] = []
    for k in range(n_sources):
        wav = overlap_add(source_frames[k], frame_len, hop_len, T)
        sources_raw.append(wav)

    sources, energy_distribution = _postprocess_sources(sources_raw, keep_top_n=2)

    avg_rank = int(np.round(np.mean(smooth_ranks))) if smooth_ranks else 1
    if singular_bank:
        first10 = np.zeros(10, dtype=np.float64)
        for s in singular_bank:
            take = min(10, len(s))
            first10[:take] += s[:take]
        first10 /= max(1, len(singular_bank))
    else:
        first10 = np.zeros(10, dtype=np.float64)
    if first10[0] > 0:
        first10 = first10 / first10[0]

    diagnostics = {
        "rank_trace": [int(v) for v in smooth_ranks],
        "rank_trace_raw": [int(v) for v in raw_ranks],
        "rank_per_frame": [int(v) for v in smooth_ranks],
        "rank_trace_domain": "time_frames",
        "first_10_singular_values": [float(v) for v in first10.tolist()],
        "singular_value_spectrum": [float(v) for v in first10.tolist()],
        "energy_distribution": energy_distribution,
        "estimated_sources": avg_rank,
    }
    return sources, diagnostics


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
        sources, diagnostics = _svd_frequency_domain(prep, cfg)
    else:
        sources, diagnostics = _svd_time_domain(prep, cfg)

    elapsed = time.perf_counter() - t0
    rtf = elapsed / prep.duration_sec if prep.duration_sec > 0 else None

    rank_trace = diagnostics.get("rank_trace", [])
    avg_rank = int(diagnostics.get("estimated_sources", 1))

    return SeparationResult(
        method="svd",
        separated_channels=sources,
        sr=prep.sr,
        rtf=rtf,
        metadata={
            "tau": cfg.tau,
            "frequency_domain": cfg.frequency_domain,
            "wiener": cfg.use_wiener,
            "estimated_sources": avg_rank,
            "rank_trace": rank_trace,
            "rank_per_frame": diagnostics.get("rank_per_frame", rank_trace),
            "rank_trace_raw": diagnostics.get("rank_trace_raw", []),
            "rank_trace_domain": diagnostics.get("rank_trace_domain"),
            "first_10_singular_values": diagnostics.get("first_10_singular_values", []),
            "singular_value_spectrum": diagnostics.get("singular_value_spectrum", []),
            "energy_distribution": diagnostics.get("energy_distribution", []),
            "rank_trace_points": len(rank_trace),
            "n_sources_returned": len(sources),
            "processing_time_sec": round(elapsed, 4),
        },
    )