"""Fine-tune the pretrained XRV DenseNet on a CSV manifest.

CSV format:
image_path,Pneumonia,Edema,...
images/example.png,1,0,...
"""
import argparse
import json
from pathlib import Path

import torch
import torchxrayvision as xrv
from torch.utils.data import DataLoader, random_split

from multimodal_diagnosis.data.xray_dataset import XrayCSVDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Manifest CSV")
    parser.add_argument("--image-root", default="", help="Prefix for image_path values")
    parser.add_argument("--output", default="artifacts/xray-finetuned.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1")
    torch.manual_seed(args.seed)
    dataset = XrayCSVDataset(args.csv, args.image_root)
    if len(dataset) < 2:
        raise ValueError("At least two rows are required for train/validation split")
    validation_size = max(1, int(len(dataset) * args.validation_split))
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=args.batch_size)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    if dataset.labels != list(model.pathologies):
        raise ValueError(
            "CSV label columns must exactly match torchxrayvision pathologies, "
            f"in this order: {list(model.pathologies)}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss(reduction="none")

    best_validation_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for images, targets, mask in train_loader:
            images, targets, mask = images.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            losses = loss_fn(model(images), targets)
            loss = losses.masked_select(mask).mean()
            loss.backward()
            optimizer.step()
        validation_loss = evaluate(model, validation_loader, loss_fn, device)
        print(f"epoch={epoch + 1} validation_loss={validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(args.output, model, dataset.labels, best_validation_loss)


def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for images, targets, mask in loader:
            values = loss_fn(model(images.to(device)), targets.to(device))
            losses.extend(values.masked_select(mask.to(device)).cpu().tolist())
    if not losses:
        raise ValueError("Validation set contains no labeled values")
    return sum(losses) / len(losses)


def save_checkpoint(path: str, model, labels: list[str], validation_loss: float) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "pathologies": labels,
            "validation_loss": validation_loss,
            "architecture": "densenet121",
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"pathologies": labels, "validation_loss": validation_loss}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
