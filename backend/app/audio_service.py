"""
backend/app/audio_service.py

Audio ingestion & preprocessing layer for the multi-speaker speech-separation
pipeline. Downstream stages (beamforming → diarization → separation →
enhancement) consume the InputRecord from the shared InputStore via input_id.
"""

from __future__ import annotations

import base64
import io
import logging
import zipfile
from dataclasses import asdict, dataclass, field
from typing import Iterable
from uuid import uuid4

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pydub import AudioSegment

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".wav", ".mp3", ".ogg", ".webm", ".flac", ".aiff",
})
TARGET_SR: int = 16_000
MAX_WAVEFORM_POINTS: int = 4_000
MAX_ZIP_FILES: int = 32
SPECTROGRAM_DPI: int = 120


# ---------------------------------------------------------------------------
# Shared store (consumed by downstream pipeline stages)
# ---------------------------------------------------------------------------

@dataclass
class InputRecord:
    """Raw audio kept in memory for downstream stages to consume by input_id."""
    X_raw: np.ndarray   # shape (channels, samples), float32
    sr: int
    source: str


class InputStore:
    """
    In-process key-value store. Downstream stages (beamforming, separation,
    etc.) call .get(input_id) to retrieve the audio matrix without re-uploading.
    """

    def __init__(self) -> None:
        self._data: dict[str, InputRecord] = {}

    def put(self, record: InputRecord) -> str:
        key = str(uuid4())
        self._data[key] = record
        return key

    def get(self, input_id: str) -> InputRecord | None:
        return self._data.get(input_id)

    def delete(self, input_id: str) -> bool:
        return self._data.pop(input_id, None) is not None

    def __len__(self) -> int:
        return len(self._data)


store = InputStore()


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class AudioInputResponse:
    input_id: str
    source: str
    sample_rate: int
    channels: int
    samples: int
    duration_sec: float
    # Signal metrics (None when not computable)
    rms_db: float | None
    peak_db: float | None
    snr_db: float | None
    crest_factor_db: float | None
    # Frontend rendering payloads
    waveform: list[float] = field(default_factory=list)
    spectrogram_png_base64: str = ""
    # Source metadata
    channel_files: list[str] = field(default_factory=list)
    all_zip_audio_files: list[str] = field(default_factory=list)
    selected_zip_audio_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# DSP utilities
# ---------------------------------------------------------------------------

def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return np.mean(samples, axis=1)


def align_signals(signals: list[np.ndarray]) -> np.ndarray:
    if not signals:
        return np.empty((0, 0), dtype=np.float32)
    min_len = min(len(s) for s in signals)
    return np.asarray([s[:min_len] for s in signals], dtype=np.float32)


def apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples <= 0:
        return signal.copy()
    if delay_samples >= len(signal):
        return np.zeros_like(signal)
    out = np.zeros_like(signal)
    out[delay_samples:] = signal[:-delay_samples]
    return out


def apply_processing(X_raw: np.ndarray, normalize: bool, noise_level: float) -> np.ndarray:
    X = np.asarray(X_raw, dtype=np.float32).copy()
    if normalize:
        peak = np.max(np.abs(X))
        if peak > 1e-8:
            X /= peak
    if noise_level > 0:
        X += np.random.randn(*X.shape).astype(np.float32) * float(noise_level)
    return X


# ---------------------------------------------------------------------------
# Signal metrics
# ---------------------------------------------------------------------------

def _to_db(value: float) -> float | None:
    return float(20 * np.log10(value)) if value > 0 else None


def compute_metrics(signal: np.ndarray) -> dict[str, float | None]:
    if signal.size == 0:
        return {"rms_db": None, "peak_db": None, "snr_db": None, "crest_factor_db": None}

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest = (peak / rms) if rms > 1e-8 else None

    # Crude SNR: compare overall RMS to the quietest 10 % of 20 frames
    frame_sz = max(1, len(signal) // 20)
    frame_rms = [
        float(np.sqrt(np.mean(signal[i:i + frame_sz] ** 2)))
        for i in range(0, len(signal) - frame_sz, frame_sz)
    ]
    noise_floor = float(np.percentile(frame_rms, 10)) if frame_rms else 0.0
    snr = (rms / noise_floor) if noise_floor > 1e-8 else None

    return {
        "rms_db": _to_db(rms),
        "peak_db": _to_db(peak),
        "snr_db": _to_db(snr) if snr is not None else None,
        "crest_factor_db": _to_db(crest) if crest is not None else None,
    }


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio_from_bytes(
    filename: str,
    file_bytes: bytes,
    target_sr: int = TARGET_SR,
    fast_mode: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Decode audio from raw bytes into a mono float32 array at target_sr.
    Uses pydub for compressed formats (mp3/ogg/webm) and soundfile otherwise.
    """
    lower = filename.lower()
    use_pydub = any(lower.endswith(ext) for ext in (".mp3", ".webm", ".ogg"))

    if use_pydub:
        seg = AudioSegment.from_file(io.BytesIO(file_bytes))
        raw = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            raw = to_mono(raw.reshape(-1, seg.channels))
        max_val = float(2 ** (8 * seg.sample_width - 1))
        if max_val > 0:
            raw /= max_val
        sr = int(seg.frame_rate)
    else:
        raw, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        sr = int(sr)
        raw = to_mono(raw)

    if sr != target_sr:
        res_type = "kaiser_fast" if fast_mode else "kaiser_best"
        raw = librosa.resample(raw, orig_sr=sr, target_sr=target_sr, res_type=res_type)

    return np.asarray(raw, dtype=np.float32), target_sr


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def downsample_waveform(signal: np.ndarray, max_points: int = MAX_WAVEFORM_POINTS) -> list[float]:
    if len(signal) <= max_points:
        return signal.astype(np.float32).tolist()
    idx = np.linspace(0, len(signal) - 1, max_points, dtype=np.int32)
    return signal[idx].astype(np.float32).tolist()


def render_spectrogram(signal: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 3))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax)
    ax.set_title("Log-frequency spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SPECTROGRAM_DPI)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------

def build_response(
    X_raw: np.ndarray,
    sr: int,
    source: str,
    normalize: bool,
    noise_level: float,
    channel_files: list[str] | None = None,
    all_zip_audio_files: list[str] | None = None,
    selected_zip_audio_files: list[str] | None = None,
) -> AudioInputResponse:
    X = apply_processing(X_raw, normalize=normalize, noise_level=noise_level)
    ch0 = X[0] if X.shape[0] > 0 else np.zeros(1, dtype=np.float32)

    input_id = store.put(InputRecord(X_raw=X_raw, sr=sr, source=source))
    duration = round(float(X.shape[1] / sr), 6) if sr and X.shape[1] else 0.0
    metrics = compute_metrics(ch0)

    return AudioInputResponse(
        input_id=input_id,
        source=source,
        sample_rate=sr,
        channels=int(X.shape[0]),
        samples=int(X.shape[1]),
        duration_sec=duration,
        rms_db=metrics["rms_db"],
        peak_db=metrics["peak_db"],
        snr_db=metrics["snr_db"],
        crest_factor_db=metrics["crest_factor_db"],
        waveform=downsample_waveform(ch0),
        spectrogram_png_base64=render_spectrogram(ch0, sr),
        channel_files=channel_files or [],
        all_zip_audio_files=all_zip_audio_files or [],
        selected_zip_audio_files=selected_zip_audio_files or [],
    )


# ---------------------------------------------------------------------------
# Public API — called by FastAPI route handlers
# ---------------------------------------------------------------------------

def process_uploaded_files(
    files: Iterable[tuple[str, bytes]],
    *,
    fast_mode: bool,
    normalize: bool,
    noise_level: float,
) -> AudioInputResponse:
    """
    Load one or more audio files and stack them as channels.
    Each file = one channel (matches real multi-mic upload workflows).
    """
    signals: list[np.ndarray] = []
    filenames: list[str] = []
    sr = TARGET_SR

    for name, data in files:
        sig, sr = load_audio_from_bytes(name, data, target_sr=TARGET_SR, fast_mode=fast_mode)
        signals.append(sig)
        filenames.append(name)

    if not signals:
        raise ValueError("No audio files were provided.")

    X_raw = align_signals(signals)
    return build_response(
        X_raw, sr,
        source="Uploaded audio",
        normalize=normalize,
        noise_level=noise_level,
        channel_files=filenames,
    )


def process_zip(
    zip_bytes: bytes,
    *,
    selected_files: list[str] | None = None,
    auto_select_n: int | None = None,
    fast_mode: bool,
    normalize: bool,
    noise_level: float,
) -> AudioInputResponse:
    """
    Extract audio from a ZIP archive, select a subset, and process them.
    Priority: explicit selected_files > auto_select_n first-N > default 2.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as arc:
        all_audio = sorted(
            n for n in arc.namelist()
            if any(n.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
            and not n.endswith("/")
        )

    if not all_audio:
        raise ValueError("No supported audio files found in the ZIP archive.")

    if len(all_audio) > MAX_ZIP_FILES:
        logger.warning("ZIP has %d audio files; capping at %d.", len(all_audio), MAX_ZIP_FILES)
        all_audio = all_audio[:MAX_ZIP_FILES]

    if selected_files:
        selected = [n for n in selected_files if n in all_audio]
        if not selected:
            raise ValueError("None of the requested files exist in the archive.")
    elif auto_select_n is not None:
        selected = all_audio[:max(1, min(auto_select_n, len(all_audio)))]
    else:
        selected = all_audio[:2]

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as arc:
        files = [(n, arc.read(n)) for n in selected]

    resp = process_uploaded_files(
        files,
        fast_mode=fast_mode,
        normalize=normalize,
        noise_level=noise_level,
    )
    resp.source = "ZIP dataset"
    resp.all_zip_audio_files = all_audio
    resp.selected_zip_audio_files = selected
    return resp


def process_live_recording(
    name: str,
    audio_bytes: bytes,
    *,
    num_mics: int,
    per_mic_delay_ms: float,
    fast_mode: bool,
    normalize: bool,
    noise_level: float,
) -> AudioInputResponse:
    """
    Simulate a multi-mic capture by applying incremental inter-channel delays
    to a single live recording.
    """
    signal, sr = load_audio_from_bytes(name, audio_bytes, target_sr=TARGET_SR, fast_mode=fast_mode)
    delay_step = int(sr * per_mic_delay_ms / 1000.0)
    channels = [apply_delay(signal, i * delay_step) for i in range(max(1, num_mics))]
    X_raw = align_signals(channels)

    resp = build_response(
        X_raw, sr,
        source="Live recording",
        normalize=normalize,
        noise_level=noise_level,
    )
    return resp


def generate_test_signal(
    *,
    duration_sec: float = 1.0,
    sr: int = TARGET_SR,
    delay_samples: int = 200,
) -> AudioInputResponse:
    """
    Two-channel 440 Hz sine tone with a fixed inter-channel delay.
    Useful for end-to-end pipeline verification without audio hardware.
    """
    n = int(max(0.25, duration_sec) * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    X_raw = np.stack([tone, apply_delay(tone, delay_samples)])

    return build_response(
        X_raw, sr,
        source="Synthetic test signal",
        normalize=True,
        noise_level=0.0,
    )