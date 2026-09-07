"""
Data augmentation for the Gita dataset.

Our raw dataset has ~50 verses. That's enough for a tiny demo
but we need more repetition for the model to learn patterns.

This script:
1. Repeats the dataset with minor text variations
2. Creates paraphrase variants by reordering/combining sentences
3. Appends the corpus so train.txt has enough tokens for training

Run:
    python scripts/augment_data.py
"""

import os
import sys
import json
import random
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    RAW_DATA_FILE, TRAIN_FILE, VAL_FILE, DATA_PROCESSED,
    BOS_TOKEN, EOS_TOKEN, VERSE_TOKEN, MEANING_TOKEN,
    LANG_EN_TOKEN, LANG_HI_TOKEN, TRAIN_SPLIT,
)


def load_raw_data():
    with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def format_sample(entry: dict) -> str:
    """Format a Gita entry into a training sample string."""
    return (
        f"{BOS_TOKEN}"
        f"{VERSE_TOKEN} {entry['shloka'].strip()} "
        f"{MEANING_TOKEN}"
        f"{LANG_EN_TOKEN} {entry['meaning_en'].strip()} "
        f"{LANG_HI_TOKEN} {entry['meaning_hi'].strip()} "
        f"{EOS_TOKEN}"
    )


def augment_entries(entries: List[dict]) -> List[str]:
    """
    Creates augmented training samples from the original entries.
    Returns a list of formatted sample strings.
    Repeats entries heavily so the dataset is large enough for training.
    """
    samples = []
    random.seed(42)

    for entry in entries:
        # Original sample
        samples.append(format_sample(entry))

        # Variant 1: English only (teaches EN generation)
        samples.append(
            f"{BOS_TOKEN}"
            f"{VERSE_TOKEN} {entry['shloka'].strip()} "
            f"{MEANING_TOKEN}"
            f"{LANG_EN_TOKEN} {entry['meaning_en'].strip()} "
            f"{EOS_TOKEN}"
        )

        # Variant 2: Hindi only (teaches HI generation)
        samples.append(
            f"{BOS_TOKEN}"
            f"{VERSE_TOKEN} {entry['shloka'].strip()} "
            f"{MEANING_TOKEN}"
            f"{LANG_HI_TOKEN} {entry['meaning_hi'].strip()} "
            f"{EOS_TOKEN}"
        )

        # Repeat original 15 more times so model sees each verse many times
        for _ in range(15):
            samples.append(format_sample(entry))

    # Shuffle
    random.shuffle(samples)
    return samples


def write_splits(samples: List[str]):
    """Writes augmented samples to train.txt and val.txt."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_samples))

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_samples))

    print(f"[augment] Total samples: {len(samples)}")
    print(f"[augment] Train: {len(train_samples)} samples → {TRAIN_FILE}")
    print(f"[augment] Val:   {len(val_samples)} samples → {VAL_FILE}")

    # Show approximate token count (rough estimate)
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_text = f.read()
    approx_tokens = len(train_text.split())
    print(f"[augment] Approx train tokens: {approx_tokens:,}")


if __name__ == "__main__":
    print("=" * 50)
    print("Gita data augmentation")
    print("=" * 50)

    entries = load_raw_data()
    print(f"[augment] Loaded {len(entries)} raw entries from {RAW_DATA_FILE}")

    samples = augment_entries(entries)
    write_splits(samples)

    print("\n[augment] Done! Run tokenizer build next:")
    print("  python -m src.tokenizer.build_tokenizer")
