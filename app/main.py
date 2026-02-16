"""FastAPI app for voice verification."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.verifier import Verifier

verifier: Verifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ECAPA model once at startup."""
    global verifier
    verifier = Verifier()
    verifier.load_model()
    yield
    verifier = None


app = FastAPI(title="Voize Demo", lifespan=lifespan)


def get_verifier() -> Verifier:
    if verifier is None:
        raise HTTPException(500, "Model not loaded")
    return verifier


@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Form(0.25),
) -> dict:
    v = get_verifier()
    data1 = await file1.read()
    data2 = await file2.read()

    # ✅ DEBUG: confirm inputs are different
    import hashlib
    print("File1:", file1.filename, "bytes:", len(data1), "md5:", hashlib.md5(data1).hexdigest())
    print("File2:", file2.filename, "bytes:", len(data2), "md5:", hashlib.md5(data2).hexdigest())

    if not data1 or not data2:
        raise HTTPException(400, "Both files must be non-empty")
    try:
        similarity, same_speaker = v.verify(data1, data2, threshold=threshold)
    except Exception as e:
        raise HTTPException(400, f"Audio processing failed: {e}") from e
    return {
        "cosine_similarity": round(similarity, 4),
        "threshold": threshold,
        "same_speaker": same_speaker,
    }




INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Voize Demo - Speaker Verification</title>
</head>
<body>
  <h1>Speaker Verification</h1>
  <form id="form">
    <p>
      <label>File 1: <input type="file" name="file1" accept="audio/*" required></label>
    </p>
    <p>
      <label>File 2: <input type="file" name="file2" accept="audio/*" required></label>
    </p>
    <p>
      <label>Threshold: <input type="number" name="threshold" value="0.25" step="0.01" min="0" max="1"></label>
    </p>
    <p><button type="submit">Verify</button></p>
  </form>
  <pre id="result"></pre>
  <script>
    document.getElementById('form').onsubmit = async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      const r = await fetch('/verify', { method: 'POST', body: fd });
      const json = await r.json();
      document.getElementById('result').textContent = JSON.stringify(json, null, 2);
    };
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve simple HTML UI that calls /verify and prints JSON."""
    return INDEX_HTML
