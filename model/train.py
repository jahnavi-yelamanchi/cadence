"""Fine-tune the Cadence turn-detection model.

Supports MPS (Apple Silicon M-series) and CUDA. Falls back to CPU.

Usage:
  make train
  # or directly:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python model/train.py --epochs 10 --batch-size 16

Logs training curves to Weights & Biases (set WANDB_API_KEY or run wandb login).
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from model.cadence_model import CadenceConfig, CadenceModel, load_processor
from model.dataset import TurnDetectionDataset, collate_fn

PROCESSED_DIR = Path("data/processed")
CKPT_DIR = Path("model/checkpoints")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(train_ds: TurnDetectionDataset) -> torch.Tensor:
    """Inverse-frequency weighting to handle class imbalance."""
    labels = train_ds.df["label"].value_counts()
    total = len(train_ds.df)
    weights = torch.tensor(
        [total / labels.get("turn_end", 1), total / labels.get("mid_thought", 1)],
        dtype=torch.float,
    )
    return weights / weights.sum() * 2  # normalise to sum to 2


def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Training on device: {device}")

    wandb.init(
        project="cadence",
        config=vars(args),
        mode="online" if os.getenv("WANDB_API_KEY") else "disabled",
    )

    processor = load_processor()
    config = CadenceConfig(
        freeze_feature_extractor=True,
        freeze_transformer_layers=args.freeze_layers,
    )
    model = CadenceModel(config).to(device)

    train_ds = TurnDetectionDataset(PROCESSED_DIR / "train.parquet", processor, augment=True)
    val_ds = TurnDetectionDataset(PROCESSED_DIR / "val.parquet", processor, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=args.workers,
    )

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=warmup_steps / total_steps
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_fir = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_values)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        false_interruptions = 0  # mid_thought classified as turn_end
        mid_thought_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_values)
                val_loss += criterion(logits, labels).item()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # FIR: mid_thought (label=1) predicted as turn_end (pred=0)
                mid_mask = labels == 1
                false_interruptions += ((preds == 0) & mid_mask).sum().item()
                mid_thought_total += mid_mask.sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_fir = false_interruptions / max(mid_thought_total, 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | val_FIR={val_fir:.3f}"
        )
        wandb.log(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc, "val_FIR": val_fir}
        )

        if val_fir < best_val_fir:
            best_val_fir = val_fir
            ckpt_path = CKPT_DIR / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_fir": val_fir}, ckpt_path)
            print(f"  ✓ Saved best checkpoint (FIR={val_fir:.3f}) → {ckpt_path}")

    wandb.finish()
    print(f"\nTraining complete. Best val FIR: {best_val_fir:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cadence turn-detection model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--freeze-layers", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args)
