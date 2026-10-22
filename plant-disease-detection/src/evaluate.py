import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import get_loaders
from src.model import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loaders, _ = get_loaders(args.data_dir)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loaders[args.split]:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()

    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    out = f"outputs/{ckpt['model']}_{args.split}_confusion.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
