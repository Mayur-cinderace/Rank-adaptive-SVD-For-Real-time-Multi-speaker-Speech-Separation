# Rank-adaptive-SVD-For-Real-time-Multi-speaker-Speech-Separation

This project now uses:
- `frontend/`: Next.js UI for input capture and visualization
- `backend/`: FastAPI backend for audio loading, preprocessing, waveform/spectrogram generation

## Run Backend

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Frontend

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL="http://localhost:8000"
npm run dev
```

Open `http://localhost:3000`.

## API Endpoints

- `GET /health`
- `POST /api/input/upload-audio`
- `POST /api/input/upload-zip`
- `POST /api/input/live`
- `POST /api/input/test-signal`

## Notes

- Live recording in the browser uses `MediaRecorder` and uploads audio to backend.
- Dataset files remain under `pages/data/` for now; you can move them to a dedicated `data/` folder later if preferred.
