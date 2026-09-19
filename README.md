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

The repository includes a DenseNet-121 transfer-learning script in
[`scripts/train_densenet_kaggle.py`](./scripts/train_densenet_kaggle.py).
It uses ImageNet initialization, masked binary cross-entropy for weak labels,
subject-level split validation, mixed precision on CUDA, cosine learning-rate
decay, early stopping, and AUROC/AUPRC/F1 reporting. Upload or clone this
repository into Kaggle, enable a GPU, and install the training dependencies:

```bash
pip install -e ".[training]"
```

The default Kaggle paths target the MIMIC-CXR weak-label and image datasets
used by this project. Override them when using different Kaggle dataset slugs:

```csv
python scripts/train_densenet_kaggle.py \
  --weak-root /kaggle/input/<weak-label-dataset> \
  --image-root /kaggle/input/<image-dataset>/official_data_iccv_final
```

The weak-label directory must contain `train_image_weak_labels_20.csv`,
`validate_image_weak_labels_20.csv`, and `concept_catalog_20.csv`. Each split
must include `image`, `view`, and `subject_id` columns. The concept catalog
defines the 20 concepts; each corresponding target column is named
`<concept>__target` and may contain `0`, `1`, or an empty value/`NaN` for an
unknown label. Only AP and PA images are used.

Useful overrides for a smaller Kaggle run:

```bash
python scripts/train_densenet_kaggle.py \
  --weak-root /kaggle/input/<weak-label-dataset> \
  --image-root /kaggle/input/<image-dataset>/official_data_iccv_final \
  --output-dir /kaggle/working/densenet121_20concepts \
  --batch-size 32 \
  --epochs 5 \
  --patience 2
```

The command writes the best checkpoint, training history, validation
probabilities, and per-concept metrics to `--output-dir`:

```bash
best_densenet121_20concepts.pt
training_history.csv
concept_probabilities_validate.csv
per_concept_metrics.csv
```

Download `best_densenet121_20concepts.pt` to the Mac project (do not commit
model weights). The existing API continues to use the TorchXRayVision
checkpoint configured by `XRAY_MODEL_WEIGHTS`; the Kaggle checkpoint is a
separate 20-concept research artifact and requires an explicit inference
adapter before it can be served by the API.
