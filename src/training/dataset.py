"""
Dataset and DataLoader for the Gita SLM.

Each training example is one complete verse→meaning prompt, padded or
truncated to BLOCK_SIZE. This is better than a flat sliding window for
short structured data like Gita verse pairs — the model learns to
complete the full meaning for each verse independently.
"""

import os
import sys
from typing import Tuple, List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from config import (
    TRAIN_FILE, VAL_FILE, TOKENIZER_MODEL,
    BLOCK_SIZE, BATCH_SIZE, DEVICE,
)


class GitaDataset(Dataset):
    """
    Each line in the text file is one complete training sample.
    Tokenizes each sample individually, truncates to BLOCK_SIZE.
    Input = tokens[:-1], Target = tokens[1:] (next-token prediction).
    """

    def __init__(self, text_file: str, tokenizer: spm.SentencePieceProcessor):
        with open(text_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        self.examples: List[List[int]] = []
        for line in lines:
            ids = tokenizer.encode(line, out_type=int)
            # Need at least 2 tokens to form an (input, target) pair
            if len(ids) >= 2:
                # Truncate to BLOCK_SIZE + 1 so we can shift by 1
                ids = ids[: BLOCK_SIZE + 1]
                self.examples.append(ids)

        total_tokens = sum(len(e) for e in self.examples)
        print(f"[dataset] {os.path.basename(text_file)}: "
              f"{len(self.examples):,} samples, "
              f"{total_tokens:,} tokens")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids = self.examples[idx]
        # Pad to BLOCK_SIZE + 1 with 0 (<pad>) if shorter
        padded = ids + [0] * (BLOCK_SIZE + 1 - len(ids))
        x = torch.tensor(padded[:-1], dtype=torch.long)  # input
        y = torch.tensor(padded[1:],  dtype=torch.long)  # target
        return x, y


def get_dataloaders(
    tokenizer: spm.SentencePieceProcessor,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader)."""

    train_dataset = GitaDataset(TRAIN_FILE, tokenizer)
    val_dataset   = GitaDataset(VAL_FILE,   tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,  # MPS doesn't support pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


def load_tokenizer() -> spm.SentencePieceProcessor:
    """Loads the pre-trained SentencePiece tokenizer."""
    if not os.path.exists(TOKENIZER_MODEL):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_MODEL}.\n"
            "Run: python -m src.tokenizer.build_tokenizer"
        )
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    return sp
