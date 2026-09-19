"""Fine-tune ImageNet DenseNet-121 on MIMIC-CXR weak labels in Kaggle.

The expected weak-label input directory contains:

* ``train_image_weak_labels_20.csv``
* ``validate_image_weak_labels_20.csv``
* ``concept_catalog_20.csv``
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm.auto import tqdm

DEFAULT_WEAK_ROOT = Path("/kaggle/input/datasets/duy231/mimic-cxr-20concept-weak-labels")
DEFAULT_IMAGE_ROOT = Path(
    "/kaggle/input/datasets/simhadrisadaram/mimic-cxr-dataset/official_data_iccv_final"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-root", type=Path, default=DEFAULT_WEAK_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/densenet121_20concepts")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(path: Path, image_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"image", "view", "subject_id"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    frame = frame[frame["view"].isin(["AP", "PA"])].copy()
    frame["image_path"] = frame["image"].map(lambda value: str(image_root / value))
    return frame.reset_index(drop=True)


def keep_existing(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    paths = [Path(path) for path in frame["image_path"]]
    with ThreadPoolExecutor(max_workers=32) as pool:
        exists = list(tqdm(pool.map(Path.is_file, paths), total=len(paths), desc=f"Check {name}"))
    kept = frame.loc[exists].reset_index(drop=True)
    print(name, "| kept:", len(kept), "| missing:", len(frame) - len(kept))
    if kept.empty:
        raise ValueError(f"No existing images found for {name}")
    return kept


class CXRDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_columns: list[str], transform) -> None:
        self.paths = frame["image_path"].tolist()
        self.targets = frame[target_columns].to_numpy(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.from_numpy(self.targets[index])


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(targets)
    if not mask.any():
        raise ValueError("Batch contains no finite labels")
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    losses = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    return losses[mask].mean()


def run_epoch(loader, model, optimizer, scaler, device, use_amp: bool, training: bool) -> float:
    model.train(training)
    total_loss, total_labels = 0.0, 0
    for images, targets in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        label_count = int(torch.isfinite(targets).sum())
        if label_count == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = masked_bce_loss(model(images), targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item() * label_count
        total_labels += label_count
    if total_labels == 0:
        raise ValueError("Loader contains no finite labels")
    return total_loss / total_labels


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1 or args.num_workers < 0:
        raise ValueError("batch size, epochs, and patience must be positive; workers cannot be negative")
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = args.weak_root / "train_image_weak_labels_20.csv"
    val_csv = args.weak_root / "validate_image_weak_labels_20.csv"
    catalog_csv = args.weak_root / "concept_catalog_20.csv"
    required_paths = [train_csv, val_csv, catalog_csv]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input CSV(s): {', '.join(missing)}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")

    concepts = pd.read_csv(catalog_csv)["concept"].tolist()
    if len(concepts) != 20:
        raise ValueError(f"Expected 20 concepts, found {len(concepts)}")
    target_columns = [f"{name}__target" for name in concepts]
    train_df = keep_existing(read_split(train_csv, args.image_root), "train")
    val_df = keep_existing(read_split(val_csv, args.image_root), "validate")
    missing_targets = [
        column
        for column in target_columns
        if column not in train_df.columns or column not in val_df.columns
    ]
    if missing_targets:
        raise ValueError(f"Missing target columns: {', '.join(missing_targets)}")
    if set(train_df.subject_id) & set(val_df.subject_id):
        raise ValueError("Train and validation splits share subject IDs")

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(
        CXRDataset(train_df, target_columns, train_transform), shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        CXRDataset(val_df, target_columns, val_transform), shuffle=False, **loader_options
    )

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, 20))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_path = args.output_dir / "best_densenet121_20concepts.pt"
    history_path = args.output_dir / "training_history.csv"
    history, best_loss, bad_epochs = [], float("inf"), 0
    print("Device:", device, "| Available GPUs:", torch.cuda.device_count())

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, scaler, device, use_amp, True)
        val_loss = run_epoch(val_loader, model, optimizer, scaler, device, use_amp, False)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        pd.DataFrame(history).to_csv(history_path, index=False)
        vram_gb = torch.cuda.memory_reserved() / 1024**3 if use_amp else 0.0
        print(
            f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"{(time.time() - start) / 60:.1f} min | "
            f"RAM {psutil.Process().memory_info().rss / 1024**3:.1f} GB | "
            f"VRAM {vram_gb:.1f} GB"
        )
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "concepts": concepts, "val_loss": best_loss},
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for images, _ in tqdm(val_loader, desc="Predict validation"):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    if probabilities.shape != (len(val_df), len(concepts)):
        raise RuntimeError(f"Unexpected probability shape: {probabilities.shape}")

    metadata_columns = [
        column for column in ["split", "subject_id", "study_id", "image", "view"]
        if column in val_df.columns
    ]
    result = val_df[metadata_columns].copy()
    for index, concept in enumerate(concepts):
        result[f"{concept}__prob"] = probabilities[:, index]
    result_path = args.output_dir / "concept_probabilities_validate.csv"
    result.to_csv(result_path, index=False)

    truth = val_df[target_columns].to_numpy(np.float32)
    metric_rows = []
    for index, concept in enumerate(concepts):
        mask = np.isfinite(truth[:, index])
        y_true, y_prob = truth[mask, index].astype(int), probabilities[mask, index]
        has_two_classes = len(np.unique(y_true)) == 2
        metric_rows.append({
            "concept": concept,
            "n_labeled": int(mask.sum()),
            "n_positive": int(y_true.sum()),
            "threshold": 0.5,
            "auroc": roc_auc_score(y_true, y_prob) if has_two_classes else np.nan,
            "auprc": average_precision_score(y_true, y_prob) if has_two_classes else np.nan,
            "f1": f1_score(y_true, y_prob >= 0.5, zero_division=0) if y_true.size else np.nan,
        })
    metrics = pd.DataFrame(metric_rows)
    metrics = pd.concat([metrics, pd.DataFrame([{
        "concept": "macro_average",
        "n_labeled": int(metrics["n_labeled"].sum()),
        "n_positive": int(metrics["n_positive"].sum()),
        "threshold": 0.5,
        "auroc": metrics["auroc"].mean(),
        "auprc": metrics["auprc"].mean(),
        "f1": metrics["f1"].mean(),
    }])], ignore_index=True)
    metrics_path = args.output_dir / "per_concept_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("Best validation loss:", best_loss)
    print("Saved model:", best_path)
    print("Saved probabilities:", result_path)
    print("Saved metrics:", metrics_path)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
