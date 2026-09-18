"""Evaluate an exported checkpoint on a held-out CSV manifest."""
import argparse

import torch
import torchxrayvision as xrv
from torch.utils.data import DataLoader

from multimodal_diagnosis.data.xray_dataset import XrayCSVDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dataset = XrayCSVDataset(args.csv, args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    loss_fn = torch.nn.BCELoss()
    losses = []
    with torch.inference_mode():
        for images, targets, mask in loader:
            values = torch.nn.functional.binary_cross_entropy(
                model(images), targets, reduction="none"
            )
            losses.extend(values.masked_select(mask).tolist())
    print({"samples": len(dataset), "mean_bce": sum(losses) / len(losses)})


if __name__ == "__main__":
    main()
