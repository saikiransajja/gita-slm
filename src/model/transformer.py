"""
GPT-style Decoder-Only Transformer — built from scratch.

Architecture walkthrough:
──────────────────────────────────────────────────────────────────
  Input tokens
       │
  Token Embedding  +  Positional Embedding
       │
  ┌────┴────────────────────────────────────┐
  │  TransformerBlock × N_LAYER             │
  │  ┌──────────────────────────────────┐   │
  │  │  LayerNorm                       │   │
  │  │  MultiHeadCausalSelfAttention    │   │
  │  │  Residual connection             │   │
  │  │  LayerNorm                       │   │
  │  │  FeedForward (MLP)               │   │
  │  │  Residual connection             │   │
  │  └──────────────────────────────────┘   │
  └─────────────────────────────────────────┘
       │
  LayerNorm
       │
  Linear head → logits over vocab
       │
  Cross-entropy loss (during training)
  or softmax + sampling (during inference)
──────────────────────────────────────────────────────────────────

Key design choices:
- Pre-LayerNorm (more stable than post-LN)
- Causal (decoder) self-attention mask: tokens can only attend left
- No positional encoding formula — learned embeddings
- Tied input/output embeddings (weight tying reduces params)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, BLOCK_SIZE,
    N_EMBD, N_HEAD, N_LAYER, DROPOUT,
    DEVICE,
)


# ─────────────────────────────────────────────
# 1. Causal Self-Attention Head
# ─────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    "Causal" means each token can only attend to itself and earlier
    tokens — this is enforced by the triangular mask. This is what
    makes the model auto-regressive: it predicts the next token
    given all previous tokens.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head  # dimension per head

        # Single linear projects input to Q, K, V all at once (3× width)
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection after concatenating heads
        self.c_proj  = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of 1s
        # registered as buffer so it moves with .to(device)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (seq len), channels (n_embd)

        # Project to Q, K, V and split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # scores shape: (B, n_head, T, T)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask: fill future positions with -inf
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax over last dimension to get attention weights
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # Weighted sum of values: (B, n_head, T, head_dim)
        out = weights @ v

        # Re-assemble heads: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(out))


# ─────────────────────────────────────────────
# 2. Feed-Forward Block (MLP)
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Applied independently at each position after attention.
    Expands to 4× dimension then projects back (standard transformer ratio).
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),  # Gaussian Error Linear Unit — smoother than ReLU
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 3. Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    A single transformer layer:
      x → LayerNorm → Attention → + x  (residual)
        → LayerNorm → MLP       → + x  (residual)

    Pre-LayerNorm placement (before sublayer) is more stable during training.
    Residual connections allow gradients to flow unchanged through depth.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # attention sublayer + residual
        x = x + self.ff(self.ln2(x))    # FFN sublayer + residual
        return x


# ─────────────────────────────────────────────
# 4. Full GPT-style Language Model
# ─────────────────────────────────────────────
class GitaSLM(nn.Module):
    """
    The complete small language model for Bhagavad Gita understanding.

    Parameters (with default config):
      - Token embedding:       VOCAB_SIZE × N_EMBD   = 4000 × 128  = 512K
      - Positional embedding:  BLOCK_SIZE × N_EMBD   = 256 × 128   = 32K
      - 4 Transformer blocks:  each ~200K params      = ~800K
      - LM head:               tied to token embedding (0 extra)
      Total: ~1.3M parameters — fits easily on CPU
    """

    def __init__(
        self,
        vocab_size: int  = VOCAB_SIZE,
        block_size: int  = BLOCK_SIZE,
        n_embd: int      = N_EMBD,
        n_head: int      = N_HEAD,
        n_layer: int     = N_LAYER,
        dropout: float   = DROPOUT,
    ):
        super().__init__()
        self.block_size = block_size

        # Input: token + positional embeddings
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, dropout, block_size) for _ in range(n_layer)]
        )

        # Final layer norm before projection to vocab
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head: projects embeddings → vocab logits
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection
        # This reduces params and often improves performance
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)

        # Scale residual projections by 1/√(2 × n_layer) for stability
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module):
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,                          # (B, T) token indices
        targets: Optional[torch.Tensor] = None,     # (B, T) for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns (logits, loss).
        - During training:   targets provided → loss computed
        - During inference:  targets=None → loss=None
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block size {self.block_size}"
        )

        # Token + position embeddings
        device = idx.device
        positions = torch.arange(T, device=device)          # (T,)
        tok_emb = self.token_embedding(idx)                 # (B, T, C)
        pos_emb = self.position_embedding(positions)        # (T, C)
        x = self.dropout(tok_emb + pos_emb)                # (B, T, C)

        # Pass through transformer blocks
        x = self.blocks(x)                                  # (B, T, C)
        x = self.ln_f(x)                                    # (B, T, C)

        # Project to vocabulary
        logits = self.lm_head(x)                            # (B, T, vocab_size)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # ignore <pad> token (id=0)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,              # (B, T) context tokens
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive generation: append one token at a time.

        temperature: > 1 = more random, < 1 = more focused
        top_k: keep only top-k logits before sampling
        eos_token_id: stop early when this token is generated
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.block_size:]

            # Forward pass (no targets during inference)
            logits, _ = self(idx_cond)

            # Take logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            logits = logits / temperature

            # Top-k filtering: zero out all but top-k logits
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

            # Stop if EOS token generated
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
