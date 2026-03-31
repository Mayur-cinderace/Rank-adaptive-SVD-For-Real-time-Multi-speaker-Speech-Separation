"""
backend/app/main.py
"""

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio_service import (
    generate_test_signal,
    process_live_recording,
    process_uploaded_files,
    process_zip,
)
from .separation_routes import router as separation_router

app = FastAPI(title="Speech Separation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Separation pipeline routes (beamforming / SVD / neural / compare) ────────
app.include_router(separation_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Input routes (unchanged from v1) ─────────────────────────────────────────

@app.post("/api/input/upload-audio")
async def upload_audio(
    files: list[UploadFile] = File(...),
    fast_mode: bool = Form(True),
    normalize: bool = Form(True),
    noise_level: float = Form(0.0),
) -> dict:
    try:
        payload = [
            (f.filename or "audio.wav", await f.read())
            for f in files
        ]
        if not payload:
            raise ValueError("No files provided.")
        return process_uploaded_files(
            payload,
            fast_mode=fast_mode,
            normalize=normalize,
            noise_level=noise_level,
        ).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/input/upload-zip")
async def upload_zip(
    zip_file: UploadFile = File(...),
    selected_files_json: str | None = Form(None),
    auto_select_n: int | None = Form(None),
    fast_mode: bool = Form(True),
    normalize: bool = Form(True),
    noise_level: float = Form(0.0),
) -> dict:
    try:
        selected_files = None
        if selected_files_json:
            selected_files = json.loads(selected_files_json)
            if not isinstance(selected_files, list):
                raise ValueError("selected_files_json must be a JSON array.")

        return process_zip(
            await zip_file.read(),
            selected_files=selected_files,
            auto_select_n=auto_select_n,
            fast_mode=fast_mode,
            normalize=normalize,
            noise_level=noise_level,
        ).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/input/live")
async def live_recording(
    audio_file: UploadFile = File(...),
    num_mics: int = Form(2),
    per_mic_delay_ms: float = Form(3.0),
    fast_mode: bool = Form(True),
    normalize: bool = Form(True),
    noise_level: float = Form(0.0),
) -> dict:
    try:
        return process_live_recording(
            audio_file.filename or "live.webm",
            await audio_file.read(),
            num_mics=num_mics,
            per_mic_delay_ms=per_mic_delay_ms,
            fast_mode=fast_mode,
            normalize=normalize,
            noise_level=noise_level,
        ).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/input/test-signal")
def test_signal(
    duration_sec: float = Form(1.0),
    sr: int = Form(16000),
    delay_samples: int = Form(200),
) -> dict:
    try:
        return generate_test_signal(
            duration_sec=duration_sec,
            sr=sr,
            delay_samples=delay_samples,
        ).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc