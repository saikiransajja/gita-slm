"""
Utility script to inspect the trained model: parameter counts,
architecture summary, and a quick test generation.

Run:
    python scripts/inspect_model.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import BEST_CHECKPOINT, VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER


def inspect():
    if not os.path.exists(BEST_CHECKPOINT):
        print(f"No checkpoint found at {BEST_CHECKPOINT}")
        print("Train the model first: python -m src.training.train")
        return

    ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu")
    cfg  = ckpt["config"]

    print("=" * 50)
    print("  Gita SLM — Model Inspection")
    print("=" * 50)
    print(f"  Trained step:    {ckpt['step']}")
    print(f"  Best val loss:   {ckpt['val_loss']:.4f}")
    print()
    print("  Architecture:")
    print(f"    Vocab size:    {cfg['vocab_size']:,}")
    print(f"    Block size:    {cfg['block_size']}")
    print(f"    Embed dim:     {cfg['n_embd']}")
    print(f"    Attn heads:    {cfg['n_head']}")
    print(f"    Layers:        {cfg['n_layer']}")
    print()

    from src.model.transformer import GitaSLM
    cfg["dropout"] = 0.0  # disable dropout for inspection/inference
    model = GitaSLM(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Parameters:      {total:,}")
    print(f"  Trainable:       {trainable:,}")
    print()

    # Layer-wise breakdown
    print("  Layer breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name:25s}  {params:>10,} params")

    print("=" * 50)


if __name__ == "__main__":
    inspect()
