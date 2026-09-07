"""
Inference engine for the Gita SLM.

Loads a saved checkpoint and generates verse meanings.
Used by both the CLI and FastAPI server.
"""

import os
import sys
from typing import Optional, Tuple, Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import sentencepiece as spm

from config import (
    BEST_CHECKPOINT, TOKENIZER_MODEL, DEVICE,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_K,
    BOS_TOKEN, EOS_TOKEN, VERSE_TOKEN, MEANING_TOKEN,
    LANG_EN_TOKEN, LANG_HI_TOKEN,
    BLOCK_SIZE,
)
from src.model.transformer import GitaSLM


class GitaInferenceEngine:
    """
    Wraps model + tokenizer for easy inference.
    Loads once, generates many times.
    """

    def __init__(self, checkpoint_path: str = BEST_CHECKPOINT):
        self.device = DEVICE
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model(checkpoint_path)

        # Cache special token ids
        self.eos_id     = self.tokenizer.piece_to_id(EOS_TOKEN)
        self.meaning_id = self.tokenizer.piece_to_id(MEANING_TOKEN)
        self.en_id      = self.tokenizer.piece_to_id(LANG_EN_TOKEN)
        self.hi_id      = self.tokenizer.piece_to_id(LANG_HI_TOKEN)

        print(f"[inference] Model loaded on {self.device}")
        print(f"[inference] Parameters: {self.model.count_parameters():,}")

    def _load_tokenizer(self) -> spm.SentencePieceProcessor:
        if not os.path.exists(TOKENIZER_MODEL):
            raise FileNotFoundError(
                f"Tokenizer not found: {TOKENIZER_MODEL}\n"
                "Run: python -m src.tokenizer.build_tokenizer"
            )
        sp = spm.SentencePieceProcessor()
        sp.load(TOKENIZER_MODEL)
        return sp

    def _load_model(self, checkpoint_path: str) -> GitaSLM:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Run: python -m src.training.train"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        cfg = checkpoint["config"]
        cfg["dropout"] = 0.0  # disable dropout during inference

        model = GitaSLM(**cfg).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def build_prompt(self, verse: str) -> str:
        """
        Constructs the structured prompt for a given shloka/verse.
        The model learns to complete the <|meaning|> part.
        """
        return (
            f"{BOS_TOKEN}"
            f"{VERSE_TOKEN} {verse.strip()} "
            f"{MEANING_TOKEN}"
        )

    @torch.no_grad()
    def generate_meaning(
        self,
        verse: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float  = TEMPERATURE,
        top_k: int          = TOP_K,
        language: str       = "both",  # "en", "hi", or "both"
    ) -> Dict[str, str]:
        """
        Generate meaning for a given shloka verse.
        """
        prompt = self.build_prompt(verse)

        # Encode prompt to token ids
        input_ids = self.tokenizer.encode(prompt, out_type=int)
        input_ids = input_ids[:BLOCK_SIZE]
        prompt_len = len(input_ids)

        idx = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Generate — stop at EOS token id
        output_ids = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self.eos_id,
        )

        # Only look at newly generated tokens (after the prompt)
        generated_ids = output_ids[0].tolist()[prompt_len:]

        # Hard-stop at first EOS in generated ids
        if self.eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(self.eos_id)]

        # Decode only the generated portion
        generated_text = self.tokenizer.decode(generated_ids)
        # Parse out English and Hindi meanings from generated text
        meaning_en, meaning_hi = self._parse_output(generated_text)

        return {
            "verse":      verse.strip(),
            "meaning_en": meaning_en,
            "meaning_hi": meaning_hi,
            "raw":        generated_text,
        }

    def _parse_output(self, text: str) -> Tuple[str, str]:
        """
        Extracts English and Hindi meaning from the generated text.
        The generated portion starts right after <|meaning|> in the prompt,
        so text here begins with <|en|> ... <|hi|> ...
        """
        meaning_en = ""
        meaning_hi = ""

        # Strip any leftover special tokens and junk
        def clean(s: str) -> str:
            # Cut off at EOS if it appears as text
            for eos_variant in (EOS_TOKEN, "|endoftext|", "endoftext|"):
                if eos_variant in s:
                    s = s.split(eos_variant, 1)[0]
            for tok in (BOS_TOKEN, VERSE_TOKEN, MEANING_TOKEN,
                        LANG_EN_TOKEN, LANG_HI_TOKEN):
                s = s.replace(tok, "")
            # Remove <unk> artifacts and stray pipe sequences like "| | |"
            s = s.replace("⁇", "").replace("<unk>", "")
            # Remove isolated pipe characters left from broken special tokens
            import re
            s = re.sub(r'(\|\s*)+$', '', s)   # trailing pipes
            s = re.sub(r'(\|\s*){3,}', '', s)  # 3+ consecutive pipes anywhere
            return s.strip()

        try:
            # Extract English: between <|en|> and <|hi|>
            if LANG_EN_TOKEN in text:
                en_part = text.split(LANG_EN_TOKEN, 1)[1]
                if LANG_HI_TOKEN in en_part:
                    meaning_en = clean(en_part.split(LANG_HI_TOKEN, 1)[0])
                else:
                    meaning_en = clean(en_part)

            # Extract Hindi: after <|hi|>
            if LANG_HI_TOKEN in text:
                hi_part = text.split(LANG_HI_TOKEN, 1)[1]
                meaning_hi = clean(hi_part)

        except Exception:
            meaning_en = clean(text)

        return meaning_en, meaning_hi


# ─────────────────────────────────────────────
# Singleton loader (used by API and CLI)
# ─────────────────────────────────────────────
_engine: Optional[GitaInferenceEngine] = None


def get_engine() -> GitaInferenceEngine:
    """Returns a cached inference engine instance."""
    global _engine
    if _engine is None:
        _engine = GitaInferenceEngine()
    return _engine
