"""LSTM phân loại tin tuyển dụng: embedding học từ đầu → BiLSTM → lớp tuyến tính.

Từ là âm tiết tiếng Việt viết thường. Chuỗi dài hơn phân vị 95 của độ dài trên tập train thì bị cắt.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

_WORD = re.compile(r"\w+")
CUT_PERCENTILE = 95


def words(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def cut_length(lengths, percentile: float = CUT_PERCENTILE) -> int:
    return int(np.ceil(np.percentile(lengths, percentile)))


@dataclass
class Vocab:
    PAD = 0
    UNK = 1
    index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, texts, min_freq: int = 2, max_size: int = 50_000) -> Vocab:
        counts = Counter(w for text in texts for w in words(text))
        kept = [w for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if c >= min_freq]
        return cls({w: i + 2 for i, w in enumerate(kept[:max_size])})

    def __len__(self) -> int:
        return len(self.index) + 2

    def encode(self, text: str, max_len: int) -> tuple[list[int], int]:
        ids = [self.index.get(w, self.UNK) for w in words(text)][:max_len] or [self.UNK]
        return ids + [self.PAD] * (max_len - len(ids)), len(ids)


class LSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, n_classes: int, emb_dim: int = 128, hidden: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=Vocab.PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        return self.out(self.dropout(torch.cat([h[0], h[1]], dim=1)))
