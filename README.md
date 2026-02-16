# Voize Demo – Speaker Verification

Voice verification using SpeechBrain ECAPA-TDNN. Upload two audio files and get cosine similarity + same-speaker decision.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the HTML UI.

## API

**POST /verify** – Two audio files (multipart/form-data), optional `threshold` (default 0.25).

### Curl example

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -F "file1=@speaker1.wav" \
  -F "file2=@speaker2.wav" \
  -F "threshold=0.25"
```

Response:

```json
{"cosine_similarity": 0.7234, "threshold": 0.25, "same_speaker": true}
```

## Threshold tuning (FAR / FRR / EER)

- **Higher threshold** → fewer false accepts (lower FAR), more false rejects (higher FRR).
- **Lower threshold** → more false accepts (higher FAR), fewer false rejects (lower FRR).
- **EER** (Equal Error Rate) is the point where FAR = FRR; that threshold is often used as a default.

Tune on your evaluation set: sweep thresholds, compute FAR/FRR, and pick the one that fits your use case (e.g. EER threshold or a chosen FAR/FRR trade-off).

## Notes

- If `app/fine_tuned_speaker_verification_model_v1.pth` exists, it is loaded over the pretrained ECAPA (from your consent notebook).
- Model expects 16 kHz mono; audio is resampled and converted to mono automatically.
- First run downloads the model to `~/.cache/huggingface/` (requires write access).
- First request may be slower while the model finishes loading.
