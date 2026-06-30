"""Distributed training for Cadence on Modal (2× A10G, PyTorch DDP).

Usage:
  # 1. One-time setup
  pip install modal && modal setup

  # 2. Store your W&B key as a Modal secret
  modal secret create wandb-secret WANDB_API_KEY=<your_key>

  # 3. Full pipeline: download data + train
  modal run modal_train.py

  # 4. Pull trained checkpoint to local machine
  modal run modal_train.py::pull_checkpoint

Resume notes:
  - Data is cached in a Modal Volume (cadence-data) — re-runs skip download
  - Best checkpoint saved to /data/checkpoints/best.pt on the volume
"""

import os
import sys
from pathlib import Path

import modal

# ── Image ──────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "git")
    .pip_install(
        "torch>=2.3.0",
        "torchaudio>=2.3.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.2",
        "pyarrow>=16.0.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "wandb>=0.17.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
    )
    .add_local_python_source("data", "model")  # copy our local modules into image
)

# ── Persistent volume (data + checkpoints survive between runs) ────────────

volume = modal.Volume.from_name("cadence-data", create_if_missing=True)
REMOTE_DATA = Path("/data")

# ── App ────────────────────────────────────────────────────────────────────

app = modal.App("cadence", image=image)

# ── Data pipeline ──────────────────────────────────────────────────────────

@app.function(
    volumes={REMOTE_DATA: volume},
    timeout=3600,
    memory=16384,
)
def setup_data() -> None:
    """Download AMI, label pauses, and split — skips if already done."""
    sys.path.insert(0, "/root")

    labels_path = REMOTE_DATA / "processed" / "labels.parquet"
    train_path = REMOTE_DATA / "processed" / "train.parquet"

    if train_path.exists():
        print("Data already prepared — skipping")
        return

    # Patch paths to point at the volume
    import data.download as dl
    import data.label as lb
    import data.split as sp

    dl.RAW_DIR = REMOTE_DATA / "raw"
    lb.RAW_DIR = REMOTE_DATA / "raw"
    lb.PROCESSED_DIR = REMOTE_DATA / "processed"
    sp.PROCESSED_DIR = REMOTE_DATA / "processed"

    print("==> Downloading AMI...")
    dl.main(["train", "validation", "test"])

    print("==> Labeling pauses...")
    lb.main()

    print("==> Splitting...")
    sp.main()

    volume.commit()
    print("Data pipeline complete.")


# ── DDP training worker ────────────────────────────────────────────────────

def _train_worker(rank: int, world_size: int, config: dict) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import wandb
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from tqdm import tqdm

    from model.cadence_model import CadenceConfig, CadenceModel, load_processor
    from model.dataset import TurnDetectionDataset, collate_fn

    # Init distributed process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        wandb.init(
            project="cadence",
            config=config,
            mode="online" if os.getenv("WANDB_API_KEY") else "disabled",
        )

    processed_dir = Path(config["processed_dir"])
    ckpt_dir = Path(config["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor()
    model_cfg = CadenceConfig(
        freeze_feature_extractor=True,
        freeze_transformer_layers=config["freeze_layers"],
    )
    model = CadenceModel(model_cfg).to(device)
    model = DDP(model, device_ids=[rank])

    train_ds = TurnDetectionDataset(processed_dir / "train.parquet", processor, augment=True)
    val_ds = TurnDetectionDataset(processed_dir / "val.parquet", processor, augment=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], sampler=train_sampler,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"] * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )

    # Class weights from training labels
    counts = train_ds.df["label"].value_counts()
    total = len(train_ds.df)
    weights = torch.tensor(
        [total / counts.get("turn_end", 1), total / counts.get("mid_thought", 1)],
        dtype=torch.float,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights / weights.sum() * 2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * config["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=total_steps, pct_start=0.1
    )

    best_val_fir = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"[GPU {rank}] Epoch {epoch:02d}/{config['epochs']}",
            disable=(rank != 0),
        )
        for batch in pbar:
            x = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(x)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validate on rank 0 only
        dist.barrier()
        if rank == 0:
            model.eval()
            val_loss, correct, total_n = 0.0, 0, 0
            false_interruptions, mid_thought_total = 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input_values"].to(device)
                    lbl = batch["labels"].to(device)
                    logits = model.module(x)
                    val_loss += criterion(logits, lbl).item()
                    preds = logits.argmax(dim=-1)
                    correct += (preds == lbl).sum().item()
                    total_n += lbl.size(0)
                    mid_mask = lbl == 1
                    false_interruptions += ((preds == 0) & mid_mask).sum().item()
                    mid_thought_total += mid_mask.sum().item()

            val_loss /= len(val_loader)
            val_acc = correct / total_n
            val_fir = false_interruptions / max(mid_thought_total, 1)

            print(
                f"Epoch {epoch:02d}/{config['epochs']} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"acc={val_acc:.3f} | FIR={val_fir:.3f}"
            )
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "val_acc": val_acc, "val_FIR": val_fir})

            if val_fir < best_val_fir:
                best_val_fir = val_fir
                torch.save(
                    {"epoch": epoch, "model_state": model.module.state_dict(), "val_fir": val_fir},
                    ckpt_dir / "best.pt",
                )
                print(f"  ✓ New best checkpoint (FIR={val_fir:.3f})")

        dist.barrier()

    if rank == 0:
        wandb.finish()

    dist.destroy_process_group()


# ── Modal training function ────────────────────────────────────────────────

@app.function(
    gpu="A10G:2",
    volumes={REMOTE_DATA: volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret", required=False)],
    memory=32768,
)
def train(
    epochs: int = 10,
    batch_size: int = 32,  # larger batch on GPU
    lr: float = 2e-5,
    freeze_layers: int = 8,
) -> None:
    import torch.multiprocessing as mp

    world_size = 2  # 2× A10G

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "freeze_layers": freeze_layers,
        "processed_dir": str(REMOTE_DATA / "processed"),
        "ckpt_dir": str(REMOTE_DATA / "checkpoints"),
        "world_size": world_size,
    }

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(_train_worker, args=(world_size, config), nprocs=world_size, join=True)

    volume.commit()
    print(f"Training complete. Checkpoint at {REMOTE_DATA}/checkpoints/best.pt")


# ── Pull checkpoint back to local machine ─────────────────────────────────

@app.local_entrypoint()
def main() -> None:
    print("Step 1: Setting up data on Modal volume...")
    setup_data.remote()

    print("\nStep 2: Training with DDP on 2× A10G...")
    train.remote()

    print("\nDone! Run `modal run modal_train.py::pull_checkpoint` to download best.pt")


@app.function(volumes={REMOTE_DATA: volume})
def _read_checkpoint() -> bytes:
    ckpt_path = REMOTE_DATA / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No checkpoint found — run training first")
    return ckpt_path.read_bytes()


@app.local_entrypoint()
def pull_checkpoint() -> None:
    """Download best.pt from Modal volume to local model/checkpoints/."""
    local_dir = Path("model/checkpoints")
    local_dir.mkdir(parents=True, exist_ok=True)
    data = _read_checkpoint.remote()
    out = local_dir / "best.pt"
    out.write_bytes(data)
    print(f"Downloaded checkpoint → {out} ({len(data) / 1e6:.1f} MB)")
