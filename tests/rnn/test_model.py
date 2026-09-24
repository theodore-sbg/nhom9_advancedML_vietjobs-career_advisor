import numpy as np
import pandas as pd
import pytest
import torch

from career_advisor.rnn.model import LSTMClassifier, Vocab, cut_length, words
from career_advisor.rnn.train import classification_report, posting_texts, tfidf_logreg, train_lstm


def test_words_lowercases_and_splits_syllables():
    assert words("Kế Toán, Excel!") == ["kế", "toán", "excel"]


def test_cut_length_is_the_95th_percentile_of_lengths():
    lengths = list(range(1, 101))

    assert cut_length(lengths) == int(np.ceil(np.percentile(lengths, 95)))


def test_vocab_keeps_frequent_words_and_maps_rare_ones_to_unk():
    vocab = Vocab.build(["kế toán kế toán", "kế excel"], min_freq=2)

    ids, length = vocab.encode("kế toán python", max_len=5)

    assert ids == [vocab.index["kế"], vocab.index["toán"], Vocab.UNK, Vocab.PAD, Vocab.PAD]
    assert length == 3


def test_vocab_truncates_to_max_len():
    vocab = Vocab.build(["a b c d e"], min_freq=1)

    ids, length = vocab.encode("a b c d e", max_len=3)

    assert len(ids) == 3 and length == 3


def test_forward_pass_gives_one_logit_per_class():
    model = LSTMClassifier(vocab_size=50, n_classes=4, emb_dim=8, hidden=6)
    x = torch.randint(1, 50, (3, 7))
    lengths = torch.tensor([7, 4, 1])

    logits = model(x, lengths)

    assert logits.shape == (3, 4)


def test_padding_does_not_change_the_output():
    torch.manual_seed(0)
    model = LSTMClassifier(vocab_size=20, n_classes=2, emb_dim=4, hidden=3).eval()
    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])

    a = model(short, torch.tensor([3]))
    b = model(padded, torch.tensor([3]))

    assert torch.allclose(a, b, atol=1e-6)


TOY = [("kế toán thuế sổ sách", 0), ("lập trình python java", 1)] * 30


def test_lstm_learns_a_separable_toy_task_and_is_deterministic():
    texts, labels = [t for t, _ in TOY], np.array([y for _, y in TOY])

    first = train_lstm(texts, labels, texts, labels, n_classes=2, epochs=15, device="cpu", seed=1)
    second = train_lstm(texts, labels, texts, labels, n_classes=2, epochs=15, device="cpu", seed=1)

    assert (first.predict(texts) == labels).all()
    assert first.history == second.history


def test_tfidf_logreg_predicts_the_toy_task():
    texts, labels = [t for t, _ in TOY], np.array([y for _, y in TOY])

    preds, c = tfidf_logreg(texts, labels, texts, labels, texts)

    assert (preds == labels).all() and c > 0


def test_classification_report_matches_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    y_true, y_pred = np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 0])

    report = classification_report(y_true, y_pred, labels=["a", "b", "c"])

    assert report["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert report["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert report["confusion"] == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]


def test_posting_texts_skip_benefits_and_raw_title():
    postings = pd.DataFrame(
        {
            "job_title": ["Kế toán (Lương 15tr)"],
            "job_title_norm": ["kế toán"],
            "description": ["Làm sổ sách"],
            "requirements_text": ["Biết Excel"],
            "benefits": ["Lương 15 triệu"],
        }
    )

    text = posting_texts(postings).iloc[0]

    assert "kế toán" in text and "Excel" in text and "15" not in text
