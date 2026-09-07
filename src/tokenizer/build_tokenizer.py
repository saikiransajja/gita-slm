"""
Builds a SentencePiece BPE tokenizer on the Gita training corpus.

Why SentencePiece BPE?
- Handles Sanskrit transliteration + English + Hindi (Devanagari) in one vocab
- Sub-word tokenization: rare words are split into known pieces
- Compact vocab (4000 tokens) suits a small model

Run:
    python -m src.tokenizer.build_tokenizer
"""

import os
import sys
import json
from typing import List

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sentencepiece as spm

from config import (
    RAW_DATA_FILE,
    DATA_PROCESSED,
    TOKENIZER_MODEL,
    VOCAB_SIZE,
    BOS_TOKEN, EOS_TOKEN, VERSE_TOKEN, MEANING_TOKEN,
    LANG_EN_TOKEN, LANG_HI_TOKEN,
    TRAIN_FILE, VAL_FILE, TRAIN_SPLIT,
)


# ─────────────────────────────────────────────
# Step 1: Build training corpus text file
# ─────────────────────────────────────────────
def build_corpus(raw_data_path: str, corpus_path: str) -> List[dict]:
    """
    Converts gita.json into a plain text corpus for tokenizer training
    and returns the parsed entries.
    """
    with open(raw_data_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for e in entries:
            # Write all text fields so the tokenizer sees all vocabulary
            f.write(e["shloka"] + "\n")
            f.write(e["meaning_en"] + "\n")
            f.write(e["meaning_hi"] + "\n")

    print(f"[tokenizer] Corpus written: {corpus_path} ({len(entries)} entries)")
    return entries


# ─────────────────────────────────────────────
# Step 2: Train SentencePiece tokenizer
# ─────────────────────────────────────────────
def train_tokenizer(corpus_path: str):
    """Trains a BPE tokenizer and saves .model + .vocab files."""

    model_prefix = TOKENIZER_MODEL.replace(".model", "")

    # Special tokens to add to vocabulary
    special_tokens = ",".join([
        BOS_TOKEN, EOS_TOKEN,
        VERSE_TOKEN, MEANING_TOKEN,
        LANG_EN_TOKEN, LANG_HI_TOKEN,
    ])

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,  # high coverage for Hindi Devanagari
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        user_defined_symbols=f"{VERSE_TOKEN},{MEANING_TOKEN},{LANG_EN_TOKEN},{LANG_HI_TOKEN}",
    )
    print(f"[tokenizer] Model saved: {TOKENIZER_MODEL}")


# ─────────────────────────────────────────────
# Step 3: Build train/val split text files
# ─────────────────────────────────────────────
def build_train_val_split(entries: List[dict]):
    """
    Formats entries as structured prompt-response pairs and splits
    into train.txt and val.txt.

    Format of each example:
        <|startoftext|><|verse|> {shloka} <|meaning|><|en|> {english} <|hi|> {hindi} <|endoftext|>
    """
    samples = []
    for e in entries:
        sample = (
            f"{BOS_TOKEN}"
            f"{VERSE_TOKEN} {e['shloka'].strip()} "
            f"{MEANING_TOKEN}"
            f"{LANG_EN_TOKEN} {e['meaning_en'].strip()} "
            f"{LANG_HI_TOKEN} {e['meaning_hi'].strip()} "
            f"{EOS_TOKEN}"
        )
        samples.append(sample)

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * TRAIN_SPLIT)
    train_samples = samples[:split_idx]
    val_samples   = samples[split_idx:]

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_samples))

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_samples))

    print(f"[tokenizer] Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    print(f"[tokenizer] Train file: {TRAIN_FILE}")
    print(f"[tokenizer] Val file:   {VAL_FILE}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    corpus_path = os.path.join(DATA_PROCESSED, "corpus.txt")

    print("=" * 50)
    print("Building Gita tokenizer")
    print("=" * 50)

    entries = build_corpus(RAW_DATA_FILE, corpus_path)
    train_tokenizer(corpus_path)
    # NOTE: train.txt / val.txt are written by scripts/augment_data.py
    # Run that AFTER this script so augmented splits are not overwritten.

    print("\n[tokenizer] Done!")
    print("  Next step: python scripts/augment_data.py")
