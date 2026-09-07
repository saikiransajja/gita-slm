"""
Training loop for the Gita SLM.

What happens here:
  1. Load tokenizer + datasets
  2. Instantiate the GitaSLM model
  3. Run the AdamW optimizer with cosine LR decay + linear warmup
  4. Evaluate on validation set every EVAL_INTERVAL steps
  5. Save best checkpoint (by val loss) and latest checkpoint
  6. Log everything to CSV

Run:
    python -m src.training.train
"""

import os
import sys
import csv
import math
import time
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS,
    LEARNING_RATE, LR_DECAY, WARMUP_ITERS, GRAD_CLIP,
    BEST_CHECKPOINT, LATEST_CHECKPOINT, TRAINING_LOG,
    N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE, VOCAB_SIZE,
    CHECKPOINTS, LOGS,
)
from src.model.transformer import GitaSLM
from src.training.dataset import get_dataloaders, load_tokenizer


# ─────────────────────────────────────────────
# Learning rate schedule: warmup + cosine decay
# ─────────────────────────────────────────────
def get_lr(step: int) -> float:
    """
    Linear warmup from 0 → LEARNING_RATE over WARMUP_ITERS steps,
    then cosine decay from LEARNING_RATE → LEARNING_RATE/10.
    """
    if not LR_DECAY:
        return LEARNING_RATE

    # Linear warmup phase
    if step < WARMUP_ITERS:
        return LEARNING_RATE * step / WARMUP_ITERS

    # Cosine decay phase
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    # Decay to 10% of the peak LR
    return LEARNING_RATE * (0.1 + 0.9 * cosine_decay)


# ─────────────────────────────────────────────
# Validation loss estimation
# ─────────────────────────────────────────────
@torch.no_grad()
def estimate_loss(
    model: GitaSLM,
    train_loader,
    val_loader,
) -> Dict[str, float]:
    """
    Averages loss over EVAL_ITERS batches for both splits.
    Uses @no_grad to skip gradient computation (faster + less memory).
    """
    model.eval()
    results = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        count = 0
        loader_iter = iter(loader)

        for _ in range(min(EVAL_ITERS, len(loader))):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                break

            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1

        results[split] = total_loss / max(count, 1)

    model.train()
    return results


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(fresh: bool = False):
    os.makedirs(CHECKPOINTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    print("=" * 60)
    print("  Gita SLM — Training")
    print("=" * 60)
    print(f"  Device:      {DEVICE}")

    # ── 1. Load tokenizer and data ──
    tokenizer = load_tokenizer()
    train_loader, val_loader = get_dataloaders(tokenizer)

    if len(train_loader) == 0:
        print("\n[ERROR] Training dataset is too small to create any batches.")
        print("        Run the data augmentation script first:")
        print("        python scripts/augment_data.py")
        return

    # ── 2. Build model ──
    model = GitaSLM(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = model.count_parameters()
    print(f"  Parameters:  {n_params:,}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Max iters:   {MAX_ITERS}")
    print("=" * 60)

    # ── 3. Optimizer (AdamW) ──
    # Separate weight decay: apply only to 2D params (weights), not biases/norms
    decay_params    = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ── 4. CSV log setup ──
    log_file = open(TRAINING_LOG, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val_loss = float("inf")
    step = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    # Resume from latest checkpoint if it exists (unless --fresh is set)
    if not fresh and os.path.exists(LATEST_CHECKPOINT):
        print(f"\n[train] Resuming from {LATEST_CHECKPOINT}")
        ckpt = torch.load(LATEST_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Do NOT restore step — always train a full new run
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[train] Loaded weights (best val loss so far: {best_val_loss:.4f}). Starting fresh run.\n")
    elif fresh:
        print("[train] Fresh run — ignoring any existing checkpoints.\n")

    # ── 5. Training loop ──
    pbar = tqdm(total=MAX_ITERS, initial=0, desc="Training", unit="step")

    while step < MAX_ITERS:
        # ── LR update ──
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Evaluation ──
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_loader, val_loader)
            elapsed = time.time() - start_time

            pbar.set_postfix(
                train_loss=f"{losses['train']:.4f}",
                val_loss=f"{losses['val']:.4f}",
                lr=f"{lr:.2e}",
            )

            log_writer.writerow([step, losses["train"], losses["val"], lr, f"{elapsed:.1f}"])
            log_file.flush()

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": {
                            "vocab_size": VOCAB_SIZE,
                            "block_size": BLOCK_SIZE,
                            "n_embd": N_EMBD,
                            "n_head": N_HEAD,
                            "n_layer": N_LAYER,
                            "dropout": DROPOUT,
                        },
                    },
                    BEST_CHECKPOINT,
                )

        # ── Get next batch (cycle through loader) ──
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        # ── Forward + backward ──
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()

        # Gradient clipping: prevents exploding gradients
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        step += 1
        pbar.update(1)

    pbar.close()
    log_file.close()

    # ── Save latest checkpoint ──
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
            "config": {
                "vocab_size": VOCAB_SIZE,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "dropout": DROPOUT,
            },
        },
        LATEST_CHECKPOINT,
    )

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model:    {BEST_CHECKPOINT}")
    print(f"  Training log:  {TRAINING_LOG}")
    print(f"{'=' * 60}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Gita SLM")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints and start training from scratch"
    )
    args = parser.parse_args()
    train(fresh=args.fresh)
