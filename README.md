# Multimodal Diagnosis

An X-ray-first foundation for a future multimodal diagnosis system. The first
model is a pretrained `torchxrayvision` DenseNet-121 checkpoint trained on
multiple chest X-ray datasets. Clinical notes are intentionally represented as
a separate input contract so a text encoder and fusion layer can be added
without changing the X-ray API.

> **Medical safety:** this is a research/engineering starter, not a medical
> device or a diagnostic tool. Model scores must be clinically validated before
> being used for patient care.

## Project structure

```text
multimodal-diagnosis/
├── src/multimodal_diagnosis/
│   ├── api.py                 # FastAPI routes and upload validation
│   ├── config.py              # Environment-backed settings
│   ├── main.py                # ASGI entrypoint
│   ├── schemas.py             # Request/response models
│   ├── models/
│   │   └── xray.py            # Lazy pretrained X-ray model wrapper
│   └── services/
│       └── diagnosis.py       # X-ray inference and result formatting
├── tests/
│   └── test_api.py            # Contract and validation tests
├── .env.example
├── pyproject.toml
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn multimodal_diagnosis.main:app --reload
```

The first X-ray request downloads the configured pretrained checkpoint and may
take a few minutes. The model is cached by `torchxrayvision`.

## API

Health check:

```bash
curl http://localhost:8000/health
```

X-ray inference:

```bash
curl -X POST http://localhost:8000/v1/diagnose/xray \
  -F "image=@/path/to/chest-xray.png"
```

The response contains probabilities for all labels and the highest-scoring
labels:

```json
{
  "model": "densenet121-res224-all",
  "predictions": [
    {"label": "Pneumonia", "probability": 0.83, "positive": true}
  ],
  "disclaimer": "Research use only; not a medical diagnosis."
}
```

Supported image formats are those understood by Pillow (PNG, JPEG, TIFF, and
similar formats). Uploads are limited to 10 MiB by default.

## Next multimodal step

Add a clinical-notes encoder under `models/clinical_notes.py` and a fusion
service under `services/multimodal.py`. Keep the existing X-ray predictions as
an auditable modality-specific signal, then train and validate a fusion model
on a paired X-ray/notes dataset. Do not concatenate uncalibrated scores in
production without an evaluation and calibration protocol.

## Fine-tune in Colab or Kaggle

The repository includes a transfer-learning script in
[`scripts/train_xray.py`](./scripts/train_xray.py). Upload or clone this
repository into Colab/Kaggle, enable a GPU, and install the package:

```bash
pip install -e .
```

Create a manifest CSV. The first column must be `image_path`; the remaining
columns must be the TorchXRayVision pathology names, in the same order as the
pretrained model. Use `0`, `1`, or an empty value for an unknown label:

```csv
image_path,Atelectasis,Consolidation,Infiltration,...
images/a.png,0,1,,
images/b.png,1,0,1
```

Train and export the best validation checkpoint:

```bash
python scripts/train_xray.py \
  --csv /content/manifest.csv \
  --image-root /content/dataset \
  --output artifacts/xray-finetuned.pt \
  --epochs 5
```

Evaluate the exported checkpoint on a separate test manifest:

```bash
python scripts/evaluate_xray.py \
  --checkpoint artifacts/xray-finetuned.pt \
  --csv /content/test.csv \
  --image-root /content/dataset
```

Download `artifacts/xray-finetuned.pt` to the Mac project (do not commit
model weights), then set:

```bash
export XRAY_CHECKPOINT=/absolute/path/to/xray-finetuned.pt
uvicorn multimodal_diagnosis.main:app --reload
```

The API loads the exported checkpoint on the first request. If
`XRAY_CHECKPOINT` is empty, it continues to use the original pretrained
checkpoint.
