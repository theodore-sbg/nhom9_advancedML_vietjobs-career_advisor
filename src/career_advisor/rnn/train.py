"""Huấn luyện LSTM (chọn epoch theo macro-F1 trên val) và mốc TF-IDF + hồi quy logistic (chọn C trên val)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from career_advisor.config import SEED
from career_advisor.rnn.model import LSTMClassifier, Vocab, cut_length, words

BATCH = 64
LR = 1e-3
PATIENCE = 2
C_CANDIDATES = (0.3, 1.0, 3.0, 10.0)


def posting_texts(postings: pd.DataFrame) -> pd.Series:
    """Chức danh đã chuẩn hoá, mô tả, yêu cầu. Bỏ phúc lợi và chức danh gốc vì hay ghi sẵn mức lương."""
    parts = postings[["job_title_norm", "description", "requirements_text"]].fillna("")
    return parts.apply(lambda r: ". ".join(x for x in r if x), axis=1)


def classification_report(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(labels)))),
        "labels": list(labels),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
    }


def _encode(vocab: Vocab, texts, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [vocab.encode(t, max_len) for t in texts]
    return torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])


@dataclass
class TrainedLSTM:
    model: LSTMClassifier
    vocab: Vocab
    max_len: int
    device: str
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0

    @torch.no_grad()
    def predict(self, texts, batch: int = 256) -> np.ndarray:
        self.model.eval()
        x, lengths = _encode(self.vocab, list(texts), self.max_len)
        out = []
        for i in range(0, len(x), batch):
            xb, lb = x[i : i + batch], lengths[i : i + batch]
            logits = self.model(xb[:, : int(lb.max())].to(self.device), lb)
            out.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(out)


def train_lstm(
    train_texts,
    y_train,
    val_texts,
    y_val,
    n_classes: int,
    epochs: int = 10,
    device: str = "cpu",
    seed: int = SEED,
    log=None,
) -> TrainedLSTM:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_texts, val_texts = list(train_texts), list(val_texts)
    vocab = Vocab.build(train_texts)
    max_len = cut_length([len(words(t)) for t in train_texts])
    x, lengths = _encode(vocab, train_texts, max_len)
    y = torch.tensor(np.asarray(y_train))
    model = LSTMClassifier(len(vocab), n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    trained = TrainedLSTM(model, vocab, max_len, device)
    best_f1, best_state, waited = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(order), BATCH):
            idx = torch.from_numpy(order[i : i + BATCH])
            lb = lengths[idx]
            xb = x[idx][:, : int(lb.max())].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lb), y[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        val_f1 = f1_score(y_val, trained.predict(val_texts), average="macro")
        trained.history.append({"epoch": epoch, "loss": total / len(x), "val_macro_f1": float(val_f1)})
        if log:
            log(trained.history[-1])
        if val_f1 > best_f1:
            best_f1, best_state, waited = val_f1, copy.deepcopy(model.state_dict()), 0
            trained.best_epoch = epoch
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return trained


def tfidf_logreg(train_texts, y_train, val_texts, y_val, test_texts) -> tuple[np.ndarray, float]:
    """Âm tiết và cặp âm tiết, TF-IDF, hồi quy logistic. C chọn theo macro-F1 trên val."""
    vectorizer = TfidfVectorizer(
        tokenizer=words, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2,
        sublinear_tf=True, max_features=200_000,
    )  # fmt: skip
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    best = None
    for c in C_CANDIDATES:
        model = LogisticRegression(C=c, max_iter=2000).fit(x_train, y_train)
        f1 = f1_score(y_val, model.predict(x_val), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, c, model)
    _, c, model = best
    return model.predict(vectorizer.transform(test_texts)), c
