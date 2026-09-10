# AI Voice Fraud Detector — Deployment Package

Wraps your trained Wav2Vec2 spoof classifier and Detect → Verify → Protect
pipeline in a FastAPI REST service, containerized for deployment anywhere.

## 0. Put your checkpoint in place

Copy your trained model file into `checkpoints/best.pt`:

```
cp /path/to/your/downloaded/fraud_detector_best.pt checkpoints/best.pt
```

The Dockerfile bakes this into the image at build time. If you'd rather not
rebuild the image every time you retrain, uncomment the `volumes:` line in
`docker-compose.yml` and mount it instead.

## 1. Run locally first (sanity check before deploying anywhere)

```bash
cd deploy
docker compose up --build
```

Wait for `Model loaded. Ready to serve.` in the logs, then test:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/detect \
  -F "audio=@/path/to/some_clip.wav"

curl -X POST http://localhost:8000/process-call \
  -F "incoming_audio=@/path/to/some_clip.wav"
```

`/process-call` may return `CHALLENGE_REQUIRED` with a `challenge_phrase`.
In that case, re-call it with the caller's spoken response attached:

```bash
curl -X POST http://localhost:8000/process-call \
  -F "incoming_audio=@/path/to/some_clip.wav" \
  -F "response_audio=@/path/to/response_clip.wav"
```

Interactive API docs are auto-generated at `http://localhost:8000/docs`.

## 2. Where to actually deploy it

You said you're not sure — here's the honest tradeoff, since the right
answer depends mainly on **traffic pattern** and **latency tolerance**:

| Option | Best for | Why |
|---|---|---|
| **Single VM + Docker** (recommended to start) | Low/steady traffic, simplest ops | The model loads once and stays warm in memory (~1-2GB RAM, a few seconds to load). One `docker compose up -d` on any $5-20/mo VM (Hetzner, DigitalOcean, EC2 t3.medium) gets you a working service today. |
| **Cloud Run / equivalent serverless containers** | Bursty traffic, want to pay only when used | Cloud Run supports "min instances = 1" to keep the model warm and avoid cold-start reloading it on every request — set that, or you'll eat multi-second Wav2Vec2 load time per cold call. |
| **Kubernetes** | Already running k8s, need autoscaling/multi-region | Overkill just for this service alone; reach for it only if it's joining a larger existing cluster. |
| **AWS Lambda (raw, not containers)** | Not recommended here | Wav2Vec2 + torch don't fit comfortably in Lambda's package/size and cold-start constraints without real tuning. Cloud Run (container-based serverless) gets you the "pay per use" benefit without that pain. |

**Recommendation for a first deployment:** a single VM with `docker compose`
(step 1 above, just pointed at a public IP + a reverse proxy for TLS), or
Cloud Run with `min-instances=1` if you want it to scale to zero when idle
overnight. Move to Kubernetes later only if you outgrow one instance.

### Deploying to a VM
```bash
# on the VM, after installing Docker:
git clone <your-repo>   # or scp the deploy/ folder over
cd deploy
docker compose up -d --build
# put nginx or Caddy in front for TLS termination on a real domain
```

### Deploying to Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/fraud-detector
gcloud run deploy fraud-detector \
  --image gcr.io/YOUR_PROJECT/fraud-detector \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --port 8000 \
  --allow-unauthenticated=false   # lock this down; this is a fraud tool
```

## 3. Moving to real-time streaming later (phase 2)

You said REST first, streaming later — when you get there, the pieces to
add are:
- A telephony integration (e.g. **Twilio Media Streams** or a SIP trunk)
  that hands you rolling audio chunks instead of a completed file.
- A buffering layer that accumulates ~4 seconds of audio (matches
  `max_duration` used in training) before calling `pipeline.detect()`,
  since the model was trained on fixed-length clips.
- A WebSocket endpoint in `main.py` instead of (or alongside) the file
  upload endpoints, since streaming doesn't fit multipart file uploads well.

Nothing in `model.py`/`preprocess.py`/`call_pipeline.py` needs to change for
this — only the audio-ingestion layer changes; the Detect/Verify/Protect
core stays the same, which is exactly why it was kept separate from I/O
concerns in the original notebook.

## 4. Operational notes

- **Security**: `/process-call` and `/detect` should sit behind auth
  (API key or mTLS) before going anywhere near the public internet — this
  tool's output influences fraud decisions.
- **Threshold tuning**: `FAKE_PROB_THRESHOLD` (default 0.6) came from your
  notebook's `call_pipeline.py` default. Re-tune it against your
  `evaluate.py` precision/recall output for your real traffic's false
  positive/negative costs before going to production.
- **Model updates**: retrain in the notebook → download new `best.pt` →
  replace `checkpoints/best.pt` → rebuild/redeploy. Consider versioning
  checkpoints (`best-2026-09-10.pt`) so you can roll back.
