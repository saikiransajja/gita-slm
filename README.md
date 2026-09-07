# Gita SLM 🕉️
### A GPT-Style Transformer Language Model Trained on the Bhagavad Gita

A GPT-style transformer language model built from scratch in **PyTorch**, trained on Bhagavad Gita verses to generate Sanskrit shloka explanations in **English and Hindi**. The model, tokenizer, training pipeline, REST API, and CLI are all custom-built — no Hugging Face, no pre-trained weights.

**~1.3M parameters · SentencePiece BPE · FastAPI · Trains on CPU**

---

## Architecture

```
Input tokens (Sanskrit verse)
       │
  Token Embedding + Positional Embedding
       │
  ┌────┴────────────────────────────────────┐
  │  TransformerBlock × 4                   │
  │  ┌──────────────────────────────────┐   │
  │  │  LayerNorm                       │   │
  │  │  Multi-Head Causal Self-Attention │   │
  │  │  Residual                        │   │
  │  │  LayerNorm                       │   │
  │  │  Feed-Forward (MLP, 4× expand)   │   │
  │  │  Residual                        │   │
  │  └──────────────────────────────────┘   │
  └─────────────────────────────────────────┘
       │
  LayerNorm → LM Head → Logits
       │
  Softmax + Sampling → Next token
```

---

## Project Structure

```
learn-slm/
├── config.py                    # Hyperparameters and paths
├── requirements.txt
│
├── data/
│   ├── raw/gita.json            # Bhagavad Gita verses (Sanskrit + EN + HI)
│   └── processed/               # Tokenized splits
│
├── src/
│   ├── model/
│   │   └── transformer.py       # GPT-style transformer — built from scratch
│   ├── tokenizer/
│   │   └── build_tokenizer.py   # SentencePiece BPE tokenizer
│   ├── training/
│   │   ├── dataset.py           # PyTorch Dataset + DataLoader
│   │   └── train.py             # Training loop — LR schedule, gradient clipping, checkpointing
│   └── inference/
│       └── generate.py          # Inference engine
│
├── api/
│   └── server.py                # FastAPI REST API
│
├── cli/
│   └── ask_gita.py              # Interactive CLI (Rich)
│
├── scripts/
│   ├── augment_data.py          # Data augmentation — expands raw verses into training samples
│   └── inspect_model.py         # Parameter inspection
│
├── checkpoints/                 # Model weights
└── logs/                        # Training loss CSV
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Training Pipeline

```bash
# Augment dataset (~50 raw verses → 250+ training samples)
python scripts/augment_data.py

# Train SentencePiece BPE tokenizer + generate splits
python -m src.tokenizer.build_tokenizer

# Train the model
python -m src.training.train
```

```
Training: 100%|████████| 5000/5000 [18:32<00:00, train_loss=2.3412, val_loss=2.8901]
Best val loss: 2.8901
```

Checkpoints saved to `checkpoints/best_model.pt`.

---

## Inference

**CLI**
```bash
python cli/ask_gita.py                                              # Interactive
python cli/ask_gita.py "karmanye vadhikaraste ma phaleshu kadachana"
python cli/ask_gita.py --verse "yada yada hi dharmasya" --lang en
```

**API**
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/meaning \
  -H "Content-Type: application/json" \
  -d '{
    "verse": "karmanye vadhikaraste ma phaleshu kadachana",
    "language": "both",
    "temperature": 0.8
  }'
```

```json
{
  "verse": "karmanye vadhikaraste ma phaleshu kadachana",
  "meaning_en": "You have a right to perform your prescribed duty...",
  "meaning_hi": "तुम्हारा अधिकार केवल कर्म करने में है...",
  "language": "both"
}
```

Swagger UI: **http://localhost:8000/docs**

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOCAB_SIZE` | 4000 | BPE vocabulary size |
| `BLOCK_SIZE` | 256 | Context window (tokens) |
| `N_EMBD` | 128 | Embedding dimension |
| `N_HEAD` | 4 | Attention heads |
| `N_LAYER` | 4 | Transformer blocks |
| `MAX_ITERS` | 5000 | Training steps |
| `LEARNING_RATE` | 3e-4 | Peak learning rate |

---

## Implementation Details

| Component | Implementation |
|-----------|---------------|
| Causal self-attention | `src/model/transformer.py` → `CausalSelfAttention` |
| Multi-head attention | `src/model/transformer.py` → `CausalSelfAttention` |
| Residual connections | `src/model/transformer.py` → `TransformerBlock` |
| Pre-LayerNorm | `src/model/transformer.py` → `TransformerBlock` |
| Weight tying | `src/model/transformer.py` → `GitaSLM.__init__` |
| BPE tokenization | `src/tokenizer/build_tokenizer.py` |
| LR warmup + cosine decay | `src/training/train.py` → `get_lr()` |
| Gradient clipping | `src/training/train.py` → training loop |
| Top-k sampling | `src/model/transformer.py` → `generate()` |
| Temperature scaling | `src/model/transformer.py` → `generate()` |

---

## Device Support

Auto-detected — no configuration needed:
- **Apple Silicon (MPS)**
- **NVIDIA GPU (CUDA)**
- **CPU** — default fallback

---
