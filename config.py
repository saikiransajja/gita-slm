"""
Central configuration for the Gita SLM.
All hyperparameters and paths live here — change this file to experiment.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED= os.path.join(BASE_DIR, "data", "processed")
CHECKPOINTS   = os.path.join(BASE_DIR, "checkpoints")
LOGS          = os.path.join(BASE_DIR, "logs")

RAW_DATA_FILE        = os.path.join(DATA_RAW,       "gita.json")
PROCESSED_DATA_FILE  = os.path.join(DATA_PROCESSED, "gita_processed.json")
TRAIN_FILE           = os.path.join(DATA_PROCESSED, "train.txt")
VAL_FILE             = os.path.join(DATA_PROCESSED, "val.txt")
TOKENIZER_MODEL      = os.path.join(DATA_PROCESSED, "tokenizer.model")
TOKENIZER_VOCAB      = os.path.join(DATA_PROCESSED, "tokenizer.vocab")
BEST_CHECKPOINT      = os.path.join(CHECKPOINTS,    "best_model.pt")
LATEST_CHECKPOINT    = os.path.join(CHECKPOINTS,    "latest_model.pt")
TRAINING_LOG         = os.path.join(LOGS,           "training_log.csv")

# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────
VOCAB_SIZE      = 4000   # small vocab for a small model
TOKENIZER_TYPE  = "bpe"  # sentencepiece BPE

# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
# ~1M parameter GPT-style transformer
BLOCK_SIZE      = 256    # max context length (tokens)
N_EMBD          = 128    # embedding dimension
N_HEAD          = 4      # attention heads
N_LAYER         = 4      # transformer blocks
DROPOUT         = 0.1

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
BATCH_SIZE      = 16
MAX_ITERS       = 5000
EVAL_INTERVAL   = 250
EVAL_ITERS      = 50
LEARNING_RATE   = 3e-4
LR_DECAY        = True
WARMUP_ITERS    = 100
GRAD_CLIP       = 1.0
TRAIN_SPLIT     = 0.9    # 90% train, 10% validation

# Device: auto-detect
import torch
DEVICE = (
    "mps"   if torch.backends.mps.is_available()   # Apple Silicon GPU
    else "cuda" if torch.cuda.is_available()        # NVIDIA GPU
    else "cpu"
)

# ─────────────────────────────────────────────
# Inference / Generation
# ─────────────────────────────────────────────
MAX_NEW_TOKENS  = 150
TEMPERATURE     = 0.3
TOP_K           = 10

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST        = "0.0.0.0"
API_PORT        = 8000

# ─────────────────────────────────────────────
# Special tokens used in prompts
# ─────────────────────────────────────────────
BOS_TOKEN       = "<|startoftext|>"
EOS_TOKEN       = "<|endoftext|>"
VERSE_TOKEN     = "<|verse|>"
MEANING_TOKEN   = "<|meaning|>"
LANG_EN_TOKEN   = "<|en|>"
LANG_HI_TOKEN   = "<|hi|>"
