"""
backend/app/separation_routes.py

FastAPI routes for the three separation methods and the unified comparison endpoint.

Endpoints:
  POST /api/separation/beamforming   — run MVDR or DAS
  POST /api/separation/svd           — run rank-adaptive SVD
  POST /api/separation/neural        — run Conv-TasNet
  POST /api/separation/compare       — run all three and return comparison table

All endpoints take an input_id (from /api/input/*) and optional config params.
The input_id is resolved from the shared InputStore in audio_service.py.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Form, HTTPException

from .audio_service import store
from .evaluation import build_comparison_table, evaluate
from .preprocessing import prepare
from .separation_beamforming import BeamformingConfig, run_beamforming
from .separation_neural import NeuralConfig, run_neural_separation
from .separation_svd import SVDConfig, run_svd_separation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/separation", tags=["separation"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_input(input_id: str):
    record = store.get(input_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"input_id '{input_id}' not found. Run /api/input/* first.")
    return record

def _empty_result(method: str, error: str) -> dict:
    return {
        "method": method,
        "sr": None,
        "n_sources": 0,
        "channels": [],
        "metrics": {
            "stoi": None,
            "pesq": None,
            "sdr_db": None,
            "sir_db": None,
            "sar_db": None,
            "rtf": None,
        },
        "metadata": {"error": error},
    }

def _wav_b64(signal: np.ndarray, sr: int) -> str:
    """Encode a float32 array as a base64 WAV string for the frontend audio player."""
    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _waveform_preview(signal: np.ndarray, max_points: int = 2000) -> list[float]:
    """Downsample waveform for frontend chart rendering."""
    if len(signal) <= max_points:
        return signal.tolist()
    idx = np.linspace(0, len(signal) - 1, max_points, dtype=np.int32)
    return signal[idx].tolist()


def _result_to_dict(result, include_audio: bool = True) -> dict:
    """Serialise a SeparationResult to a JSON-safe dict."""
    channels_out = []
    for i, ch in enumerate(result.separated_channels):
        entry: dict = {
            "index": i,
            "waveform": _waveform_preview(ch),
            "n_samples": len(ch),
        }
        if include_audio:
            try:
                entry["wav_b64"] = _wav_b64(ch, result.sr)
            except Exception:
                entry["wav_b64"] = None
        channels_out.append(entry)

    return {
        "method": result.method,
        "sr": result.sr,
        "n_sources": len(result.separated_channels),
        "estimated_sources": result.metadata.get("estimated_sources"),
        "channels": channels_out,
        "metrics": {
            "stoi": result.stoi_scores[0] if result.stoi_scores else None,
            "pesq": result.pesq_scores[0] if result.pesq_scores else None,
            "sdr_db": result.sdr,
            "sir_db": result.sir,
            "sar_db": result.sar,
            "rtf": result.rtf,
        },
        "metadata": result.metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Beamforming endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/beamforming")
def beamforming_endpoint(
    input_id: str = Form(...),
    variant: str = Form("mvdr"),            # "mvdr" | "das"
    mic_spacing_m: float = Form(0.05),
    diagonal_loading: float = Form(1e-3),
) -> dict:
    """
    Run beamforming (MVDR or delay-and-sum) on a stored audio input.

    Uses GCC-PHAT to estimate inter-mic delays — no oracle DOA required.
    """
    record = _get_input(input_id)
    prep = prepare(record.X_raw, record.sr, do_normalize=True)

    cfg = BeamformingConfig(
        variant=variant,
        mic_spacing_m=mic_spacing_m,
        diagonal_loading=diagonal_loading,
    )
    result = run_beamforming(prep, cfg)
    result = evaluate(result, reference=prep.waveform[0], sr=prep.sr)

    return _result_to_dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# SVD endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/svd")
def svd_endpoint(
    input_id: str = Form(...),
    tau: float = Form(0.85),
    frequency_domain: bool = Form(True),
    use_wiener: bool = Form(True),
    max_sources: int = Form(4),
) -> dict:
    """
    Run rank-adaptive SVD separation.

    tau              — energy threshold for rank selection (tuned on LibriMix dev set)
    frequency_domain — True: STFT-domain SVD (better);  False: time-domain OLA variant
    use_wiener       — apply Wiener post-filter using singular value tail as noise estimate
    max_sources      — cap on number of sources to return
    """
    record = _get_input(input_id)
    prep = prepare(record.X_raw, record.sr, do_normalize=True)

    cfg = SVDConfig(
        tau=tau,
        frequency_domain=frequency_domain,
        use_wiener=use_wiener,
        max_sources=max_sources,
    )
    result = run_svd_separation(prep, cfg)
    result = evaluate(result, reference=prep.waveform[0], sr=prep.sr)

    return _result_to_dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Neural endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/neural")
def neural_endpoint(
    input_id: str = Form(...),
    model_id: str = Form("mpariente/ConvTasNet_Libri2Mix_sepclean"),
    use_gpu: bool = Form(False),
    chunk_duration_sec: float = Form(30.0),
) -> dict:
    """
    Run Conv-TasNet neural separation.

    model_id          — asteroid / HuggingFace model identifier
    use_gpu           — use CUDA if available (falls back to CPU)
    chunk_duration_sec — chunk long recordings to avoid OOM
    """
    record = _get_input(input_id)
    prep = prepare(record.X_raw, record.sr, do_normalize=True)

    cfg = NeuralConfig(
        model_id=model_id,
        use_gpu=use_gpu,
        chunk_duration_sec=chunk_duration_sec if chunk_duration_sec > 0 else None,
    )
    result = run_neural_separation(prep, cfg)
    result = evaluate(result, reference=prep.waveform[0], sr=prep.sr)

    return _result_to_dict(result)


# ─────────────────────────────────────────────────────────────────────────────
# Unified comparison endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/compare")
def compare_endpoint(
    input_id: str = Form(...),
    run_beamforming_flag: bool = Form(True),
    run_das_flag: bool = Form(False),
    run_mvdr_flag: bool = Form(True),
    run_svd_flag: bool = Form(True),
    run_neural_flag: bool = Form(False),    # off by default (needs asteroid)
    # Beamforming params (shared)
    bf_variant: str = Form("mvdr"),
    bf_mic_spacing_m: float = Form(0.05),
    bf_diagonal_loading: float = Form(1e-3),
    # SVD params
    svd_tau: float = Form(0.90),
    svd_frequency_domain: bool = Form(True),
    svd_use_wiener: bool = Form(True),
    # Neural params
    neural_model_id: str = Form("mpariente/ConvTasNet_Libri2Mix_sepclean"),
    neural_use_gpu: bool = Form(False),
) -> dict:
    """
    Run all enabled methods on the same input and return a unified comparison table.

    This is the core endpoint for the research evaluation — all three methods
    consume the same PreparedSignal, ensuring apples-to-apples comparison.
    """
    record = _get_input(input_id)

    # Single preprocessing call — shared by ALL methods
    prep = prepare(record.X_raw, record.sr, do_normalize=True)

    results = []
    per_method_outputs = {}

    # ── Beamforming ──────────────────────────────────────────────────────
    def _run_beamforming_variant(variant: str, key: str) -> None:
        try:
            bf_cfg = BeamformingConfig(
                variant=variant,
                mic_spacing_m=bf_mic_spacing_m,
                diagonal_loading=bf_diagonal_loading,
            )
            bf_result = run_beamforming(prep, bf_cfg)
            # Frontend expects method ids aligned with configured method cards.
            bf_result.method = key
            bf_result = evaluate(bf_result, reference=prep.waveform[0], sr=prep.sr)
            results.append(bf_result)
            per_method_outputs[key] = _result_to_dict(bf_result)
        except Exception as exc:
            logger.error("Beamforming (%s) failed: %s", key, exc)
            per_method_outputs[key] = _empty_result(key, str(exc))

    if run_beamforming_flag:
        if run_das_flag:
            _run_beamforming_variant("das", "das")
        if run_mvdr_flag:
            _run_beamforming_variant("mvdr", "mvdr")
        # Backward-compat fallback for older clients that only send bf_variant.
        if not run_das_flag and not run_mvdr_flag:
            key = "mvdr" if bf_variant == "mvdr" else "das"
            _run_beamforming_variant(bf_variant, key)

    # ── SVD ──────────────────────────────────────────────────────────────
    if run_svd_flag:
        try:
            svd_cfg = SVDConfig(
                tau=svd_tau,
                frequency_domain=svd_frequency_domain,
                use_wiener=svd_use_wiener,
            )
            svd_result = run_svd_separation(prep, svd_cfg)
            svd_result = evaluate(svd_result, reference=prep.mono_mix, sr=prep.sr)
            results.append(svd_result)
            per_method_outputs["svd"] = _result_to_dict(svd_result)
        except Exception as exc:
            logger.error("SVD failed: %s", exc)
            per_method_outputs["svd"] = _empty_result("svd", str(exc))

    # ── Neural ───────────────────────────────────────────────────────────
    if run_neural_flag:
        try:
            n_cfg = NeuralConfig(
                model_id=neural_model_id,
                use_gpu=neural_use_gpu,
            )
            n_result = run_neural_separation(prep, n_cfg)
            n_result = evaluate(n_result, reference=prep.mono_mix, sr=prep.sr)
            results.append(n_result)
            per_method_outputs["neural"] = _result_to_dict(n_result)
        except Exception as exc:
            logger.error("Neural failed: %s", exc)
            per_method_outputs["neural"] = _empty_result("neural", str(exc))

    return {
        "input_id": input_id,
        "n_channels": prep.n_channels,
        "duration_sec": prep.duration_sec,
        "sr": prep.sr,
        # Shared preprocessing params (same for all methods)
        "preprocessing": {
            "normalized": True,
            "n_fft": prep.n_fft,
            "hop_length": prep.hop_length,
            "win_length": prep.win_length,
            "target_sr": prep.sr,
        },
        # Flat comparison table for frontend charts
        "comparison_table": build_comparison_table(results),
        # Full per-method output (waveforms, audio, metadata)
        "methods": per_method_outputs,
    }