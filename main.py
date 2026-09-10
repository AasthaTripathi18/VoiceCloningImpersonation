"""
main.py

REST API for the AI voice fraud detector, built on the Detect -> Verify ->
Protect pipeline from the training notebook.

Endpoints:
    GET  /health                      liveness/readiness probe
    POST /detect                      upload one audio clip -> fake probability
    POST /process-call                upload incoming (+ optional response)
                                       audio -> full ALLOW/CHALLENGE/BLOCK decision

The model and ASR pipeline are loaded ONCE at process startup (not per
request) since they're expensive to initialize. This is why deployment
needs a long-lived server process (or a container that stays warm) rather
than a bare serverless function per call, unless you accept cold-start
latency on every invocation -- see the README for tradeoffs.
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from call_pipeline import CallFraudPreventionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_detector")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/best.pt")
PRETRAINED_MODEL = os.environ.get("PRETRAINED_MODEL", "facebook/wav2vec2-base")
FAKE_PROB_THRESHOLD = float(os.environ.get("FAKE_PROB_THRESHOLD", "0.6"))
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

pipeline: CallFraudPreventionPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Mount/copy your "
            f"trained best.pt there or set CHECKPOINT_PATH."
        )
    logger.info("Loading model from %s ...", CHECKPOINT_PATH)
    pipeline = CallFraudPreventionPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        pretrained_model=PRETRAINED_MODEL,
        fake_prob_threshold=FAKE_PROB_THRESHOLD,
        whisper_model_size=WHISPER_MODEL_SIZE,
    )
    logger.info("Model loaded. Ready to serve.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Voice Fraud Detector",
    version="1.0.0",
    lifespan=lifespan,
)


class DetectResponse(BaseModel):
    fake_probability: float
    risk_level: str


def _validate_and_save_upload(upload: UploadFile) -> str:
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with tmp:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_MB}MB limit.",
                    )
                tmp.write(chunk)
    except HTTPException:
        os.unlink(tmp.name)
        raise
    return tmp.name


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/detect", response_model=DetectResponse)
async def detect(audio: UploadFile = File(...)):
    """Run just the Detect layer: fake-probability for one clip."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    path = _validate_and_save_upload(audio)
    try:
        fake_prob = pipeline.detect(path)
        return DetectResponse(
            fake_probability=round(fake_prob, 4),
            risk_level="high" if fake_prob > FAKE_PROB_THRESHOLD else "low",
        )
    except Exception as e:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        os.unlink(path)


@app.post("/process-call")
async def process_call(
    incoming_audio: UploadFile = File(...),
    response_audio: UploadFile | None = File(None),
):
    """
    Run the full Detect -> Verify -> Protect pipeline.

    - incoming_audio: the caller's audio to screen.
    - response_audio: optional. Omit on the first call; if the result
      comes back as CHALLENGE_REQUIRED, play `challenge_phrase` to the
      caller via TTS, collect their spoken response, then call this
      endpoint again with response_audio attached.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    incoming_path = _validate_and_save_upload(incoming_audio)
    response_path = _validate_and_save_upload(response_audio) if response_audio else None

    try:
        result = pipeline.process_call(incoming_path, response_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception("Call pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        os.unlink(incoming_path)
        if response_path:
            os.unlink(response_path)
