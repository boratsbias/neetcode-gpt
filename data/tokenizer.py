"""Character-level tokenizer.

Minimal, dependency-free. Maps each unique character in a corpus to an integer.
Sufficient for training a small GPT on a single text file.
"""
from __future__ import annotations

from typing import Iterable, List


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_int = {c: i for i, c in enumerate(chars)}
        self.int_to_char = {i: c for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_int)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_int[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.int_to_char[int(i)] for i in ids)
