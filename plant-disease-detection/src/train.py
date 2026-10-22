import argparse
import json
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_loaders
from src.model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--model", default="mobilenet_v2",
                   choices=["mobilenet_v2", "efficientnet_b0", "resnet50"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, classes = get_loaders(args.data_dir, args.batch_size)
    model = build_model(args.model, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        print(f"[{epoch}/{args.epochs}] train {tr_acc:.3f} | val {va_acc:.3f} | {time.time()-t0:.0f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": args.model, "classes": classes,
                        "state_dict": model.state_dict()}, f"models/{args.model}_best.pt")

    with open(f"outputs/{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
