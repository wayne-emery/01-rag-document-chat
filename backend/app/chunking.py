"""Token-aware text chunking with sentence-boundary preference.

We chunk by token count using the embedder's own tokenizer so chunks always fit
within the model's max input length. We also try to break at sentence boundaries
(via a simple regex) so chunks read coherently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    text: str
    token_count: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Falls back to whole text if no boundaries found."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split `text` into chunks of at most `chunk_size` tokens with `chunk_overlap` overlap.

    Approach:
      1. Split into sentences.
      2. Greedily pack sentences into a chunk until the next sentence would overflow.
      3. When emitting a chunk, retain the trailing `chunk_overlap` tokens worth of
         sentences as the seed of the next chunk for context continuity.
      4. If a single sentence exceeds `chunk_size`, hard-split it on tokens.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    def tok_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> list[str]:
        """Emit the current chunk and return overlap seed sentences for the next."""
        nonlocal current, current_tokens
        if not current:
            return []
        text_out = " ".join(current).strip()
        chunks.append(Chunk(text=text_out, token_count=current_tokens))
        if chunk_overlap == 0:
            current = []
            current_tokens = 0
            return []
        seed: list[str] = []
        seed_tokens = 0
        for s in reversed(current):
            t = tok_len(s)
            if seed_tokens + t > chunk_overlap:
                break
            seed.insert(0, s)
            seed_tokens += t
        current = list(seed)
        current_tokens = seed_tokens
        return seed

    for sentence in sentences:
        s_tokens = tok_len(sentence)

        if s_tokens > chunk_size:
            if current:
                flush()
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            for i in range(0, len(ids), chunk_size):
                piece_ids = ids[i : i + chunk_size]
                piece_text = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
                if piece_text:
                    chunks.append(Chunk(text=piece_text, token_count=len(piece_ids)))
            current, current_tokens = [], 0
            continue

        if current_tokens + s_tokens > chunk_size:
            flush()

        current.append(sentence)
        current_tokens += s_tokens

    if current:
        flush()

    return chunks
