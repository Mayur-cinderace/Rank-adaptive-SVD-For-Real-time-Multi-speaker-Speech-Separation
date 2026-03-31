"""
backend/app/evaluation.py

Shared evaluation pipeline for all three separation methods.
Computes STOI, PESQ, SDR/SIR/SAR and RTF from a SeparationResult.

All metrics are computed against the same reference signal (the mono mix
of the original input, or a provided clean reference if available).

Metric libraries:
  - pystoi  → STOI  (0–1, higher = more intelligible)
  - pesq    → PESQ  (MOS-LQO −0.5–4.5, higher = better perceived quality)
  - mir_eval → SDR / SIR / SAR
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .preprocessing import SeparationResult, TARGET_SR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align(ref: np.ndarray, est: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate or zero-pad est to match ref length."""
    L = len(ref)
    if len(est) >= L:
        return ref, est[:L]
    return ref, np.pad(est, (0, L - len(est)))


def _safe_stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    try:
        from pystoi import stoi  # type: ignore
        ref, est = _align(ref, est)
        return float(stoi(ref.astype(np.float64), est.astype(np.float64), sr, extended=False))
    except Exception as exc:
        logger.debug("STOI failed: %s", exc)
        return None


def _safe_pesq(ref: np.ndarray, est: np.ndarray, sr: int) -> float | None:
    try:
        from pesq import pesq  # type: ignore
        ref, est = _align(ref, est)
        # PESQ requires exactly 8 kHz or 16 kHz
        mode = "wb" if sr == 16_000 else "nb"
        return float(pesq(sr, ref.astype(np.float64), est.astype(np.float64), mode))
    except Exception as exc:
        logger.debug("PESQ failed: %s", exc)
        return None


def _safe_bss_eval(
    ref: np.ndarray, est: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Returns (SDR, SIR, SAR)."""
    try:
        import mir_eval  # type: ignore
        ref, est = _align(ref, est)
        # mir_eval expects (n_sources, n_samples) matrices
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            ref[np.newaxis, :].astype(np.float64),
            est[np.newaxis, :].astype(np.float64),
        )
        return float(sdr[0]), float(sir[0]), float(sar[0])
    except Exception as exc:
        logger.debug("BSS eval failed: %s", exc)
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    result: SeparationResult,
    reference: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> SeparationResult:
    """
    Compute all metrics for a SeparationResult in-place and return it.

    Parameters
    ----------
    result    : SeparationResult from any of the three separation methods
    reference : (T,) float32 clean reference signal.
                If None, the first separated channel is used as its own
                reference (computes self-metrics — useful for development).
    sr        : sample rate (defaults to result.sr)

    Returns
    -------
    result : same object with stoi_scores, pesq_scores, sdr, sir, sar filled in
    """
    sr = sr or result.sr
    if not result.separated_channels:
        return result

    # Use the best-quality estimate (channel 0) for SDR/SIR/SAR
    primary = result.separated_channels[0]

    if reference is None:
        logger.debug(
            "evaluate(): no clean reference provided — using primary estimate as reference. "
            "Metrics will reflect separation quality relative to the mixture, not ground truth."
        )
        reference = primary

    # ── STOI (per separated channel) ──────────────────────────────────────
    stoi_scores = []
    for ch in result.separated_channels:
        s = _safe_stoi(reference, ch, sr)
        if s is not None:
            stoi_scores.append(s)
    result.stoi_scores = stoi_scores

    # ── PESQ (primary channel only — expensive) ───────────────────────────
    pesq_scores = []
    p = _safe_pesq(reference, primary, sr)
    if p is not None:
        pesq_scores.append(p)
    result.pesq_scores = pesq_scores

    # ── SDR / SIR / SAR ───────────────────────────────────────────────────
    sdr, sir, sar = _safe_bss_eval(reference, primary)
    result.sdr = sdr
    result.sir = sir
    result.sar = sar

    logger.info(
        "[%s] STOI=%.3f  PESQ=%s  SDR=%s  RTF=%s",
        result.method,
        stoi_scores[0] if stoi_scores else float("nan"),
        f"{pesq_scores[0]:.2f}" if pesq_scores else "—",
        f"{sdr:.1f} dB" if sdr is not None else "—",
        f"{result.rtf:.3f}" if result.rtf is not None else "—",
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(results: list[SeparationResult]) -> list[dict]:
    """
    Build a list of dicts suitable for JSON serialisation and frontend rendering.

    Each dict has keys:
      method, stoi, pesq, sdr, sir, sar, rtf, n_sources, metadata
    """
    rows = []
    for r in results:
        rows.append({
            "method": r.method,
            "stoi": round(r.stoi_scores[0], 4) if r.stoi_scores else None,
            "pesq": round(r.pesq_scores[0], 3) if r.pesq_scores else None,
            "sdr_db": round(r.sdr, 2) if r.sdr is not None else None,
            "sir_db": round(r.sir, 2) if r.sir is not None else None,
            "sar_db": round(r.sar, 2) if r.sar is not None else None,
            "rtf": round(r.rtf, 4) if r.rtf is not None else None,
            "n_sources": len(r.separated_channels),
            "metadata": r.metadata,
        })
    return rows