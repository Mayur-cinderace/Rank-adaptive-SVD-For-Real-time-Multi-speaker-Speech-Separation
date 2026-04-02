"""
backend/app/separation_neural.py

Neural source separation using a pre-trained Conv-TasNet from the
asteroid toolkit (Baseline 3 in the research document §10).

Design:
  - Uses the same PreparedSignal / SeparationResult schema as the other methods.
  - Takes the MONO MIX from the PreparedSignal as neural model input
    (Conv-TasNet is single-channel; it learns spectro-temporal masks internally).
  - Gracefully degrades when asteroid / torch are not installed, returning
    a clear error rather than crashing the whole pipeline.
  - Real-time factor is measured identically to the SVD method.

Model used: 'mpariente/ConvTasNet_Libri2Mix_sepclean'
  — trained on LibriMix 2-speaker clean condition (matches §10 of research doc).
  — CPU-feasible for clips up to ~30 s.

Note: asteroid may download model weights on first call (~60 MB). Subsequent
calls use the cached weights from ~/.cache/torch/hub/.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .preprocessing import PreparedSignal, SeparationResult, TARGET_SR

logger = logging.getLogger(__name__)

# Asteroid model identifier (matches research doc §10)
MODEL_ID = "mpariente/ConvTasNet_Libri2Mix_sepclean"

# Lazy-loaded to avoid import-time crashes if asteroid is missing
_model_cache: dict[str, object] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NeuralConfig:
    model_id: str = MODEL_ID
    # Use GPU if available — falls back to CPU transparently
    use_gpu: bool = False
    # Chunk the audio for long recordings to avoid OOM (None = no chunking)
    chunk_duration_sec: float | None = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (lazy, cached)
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_id: str, use_gpu: bool):
    """
    Load Conv-TasNet from asteroid hub. Cached after first call.

    Returns the model object or raises ImportError / RuntimeError.
    """
    cache_key = f"{model_id}:{'gpu' if use_gpu else 'cpu'}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        from asteroid.models import ConvTasNet  # type: ignore
        import torch  # type: ignore
    except ImportError as e:
        raise ImportError(
            "asteroid and torch are required for neural separation. "
            "Install them with: pip install asteroid torch"
        ) from e

    logger.info("Loading Conv-TasNet model '%s' (first call may download weights)...", model_id)
    try:
        model = ConvTasNet.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_id}': {exc}") from exc

    device = "cpu"
    if use_gpu:
        try:
            if torch.cuda.is_available():
                device = "cuda"
                model = model.cuda()
            else:
                logger.warning("use_gpu=True but CUDA not available — using CPU.")
        except Exception:
            pass

    model = model.eval()
    _model_cache[cache_key] = model
    logger.info("Conv-TasNet loaded on %s.", device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_inference(model, mixture: np.ndarray, use_gpu: bool) -> np.ndarray:
    """
    Run Conv-TasNet inference on a 1-D mixture.

    Returns
    -------
    sources : (n_src, T) float32
    """
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise ImportError("torch is required for neural inference.") from e

    # Model expects (batch, channels, time) = (1, 1, T)
    x = torch.from_numpy(mixture[np.newaxis, np.newaxis, :].astype(np.float32))
    if use_gpu and torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        # asteroid models return (batch, n_src, time)
        out = model(x)

    # out: (1, n_src, T) → (n_src, T) numpy
    return out[0].cpu().numpy().astype(np.float32)


def _chunk_and_run(
    model,
    mixture: np.ndarray,
    sr: int,
    chunk_sec: float,
    use_gpu: bool,
) -> np.ndarray:
    """
    Split long recordings into overlapping chunks, run inference on each,
    and stitch back together with 50 % overlap-add.

    Returns
    -------
    sources : (n_src, T) float32
    """
    chunk_len = int(chunk_sec * sr)
    hop_len = chunk_len // 2
    T = len(mixture)

    if T <= chunk_len:
        return _run_inference(model, mixture, use_gpu)

    # First pass to get n_src
    first = _run_inference(model, mixture[:chunk_len], use_gpu)
    n_src = first.shape[0]

    accumulated = np.zeros((n_src, T), dtype=np.float64)
    norm = np.zeros(T, dtype=np.float64)
    win = np.hanning(chunk_len).astype(np.float64)

    starts = list(range(0, T - chunk_len + 1, hop_len))
    if starts[-1] + chunk_len < T:
        starts.append(T - chunk_len)

    for i, start in enumerate(starts):
        chunk = mixture[start:start + chunk_len]
        if i == 0:
            sources_chunk = first
        else:
            sources_chunk = _run_inference(model, chunk, use_gpu)

        end = start + chunk_len
        for s in range(n_src):
            accumulated[s, start:end] += sources_chunk[s] * win
        norm[start:end] += win

    nz = norm > 1e-8
    for s in range(n_src):
        accumulated[s, nz] /= norm[nz]

    return accumulated.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_neural_separation(
    prep: PreparedSignal,
    cfg: NeuralConfig | None = None,
) -> SeparationResult:
    """
    Run Conv-TasNet source separation on a PreparedSignal.

    Input:  mono mix of all channels (PreparedSignal.mono_mix)
    Output: SeparationResult with n_src separated channels

    The model always receives audio at TARGET_SR (16 kHz) — guaranteed by
    the shared preprocessing pipeline.

    Parameters
    ----------
    prep : output of preprocessing.prepare()
    cfg  : NeuralConfig — defaults to CPU inference, no chunking limit

    Returns
    -------
    SeparationResult
    """
    if cfg is None:
        cfg = NeuralConfig()

    t0 = time.perf_counter()

    # Input to the neural model: mono mixture (channel mean)
    # This is exactly the same waveform produced by preprocessing.prepare()
    mixture = prep.mono_mix   # (T,) float32

    try:
        model = _load_model(cfg.model_id, cfg.use_gpu)
    except (ImportError, RuntimeError) as exc:
        logger.error("Neural model unavailable: %s", exc)
        return SeparationResult(
            method="neural",
            separated_channels=[mixture.copy(), mixture.copy()],
            sr=prep.sr,
            metadata={"error": str(exc), "model_id": cfg.model_id},
        )

    try:
        if cfg.chunk_duration_sec is not None and prep.duration_sec > cfg.chunk_duration_sec:
            sources_np = _chunk_and_run(
                model, mixture, prep.sr, cfg.chunk_duration_sec, cfg.use_gpu
            )
        else:
            sources_np = _run_inference(model, mixture, cfg.use_gpu)
    except Exception as exc:
        logger.error("Neural inference failed: %s", exc)
        return SeparationResult(
            method="neural",
            separated_channels=[mixture.copy(), mixture.copy()],
            sr=prep.sr,
            metadata={"error": str(exc), "model_id": cfg.model_id},
        )

    elapsed = time.perf_counter() - t0
    rtf = elapsed / prep.duration_sec if prep.duration_sec > 0 else None

    separated = [sources_np[k] for k in range(sources_np.shape[0])]

    return SeparationResult(
        method="neural",
        separated_channels=separated,
        sr=prep.sr,
        rtf=rtf,
        metadata={
            "model_id": cfg.model_id,
            "n_sources": len(separated),
            "input_channels": prep.n_channels,
            "processing_time_sec": round(elapsed, 4),
            "chunked": cfg.chunk_duration_sec is not None and prep.duration_sec > (cfg.chunk_duration_sec or 0),
        },
    )