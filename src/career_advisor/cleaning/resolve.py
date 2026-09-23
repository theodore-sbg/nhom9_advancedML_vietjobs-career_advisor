"""Tầng 2 và 3 của bước gộp tên kỹ năng: embedding bge-m3, rồi LLM cho vùng không chắc chắn.

- Cặp có cosine ≥ `AUTO_THRESHOLD` được gộp tự động (tầng embedding).
- Cặp có cosine trong [`LLM_LOW`, `AUTO_THRESHOLD`) và cả hai kỹ năng đủ nhiều tin thì hỏi LLM
  (tầng LLM). LLM chỉ thấy tên đại diện sau tầng embedding, và được hỏi theo lô.
Tất cả cặp được chấp nhận đi qua một union-find duy nhất, xét cặp giống nhất trước, với 3 chốt chặn:
- cụm không vượt quá `max_cluster` thành viên, để chuỗi A≈B≈C không kéo hai kỹ năng khác xa vào nhau;
- cặp cấm gộp (java và javascript…) không được nằm chung cụm, kể cả gián tiếp;
- hai tên khác số ("2d", "3d") hoặc khác ngôn ngữ ("tiếng anh", "tiếng trung") không gộp.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from career_advisor.llm import extract_json

# Ngưỡng tạm. Task 8 chọn lại trên phần dev của bộ nhãn tay.
AUTO_THRESHOLD = 0.92
LLM_LOW = 0.84
MAX_CLUSTER = 30
# Chỉ hỏi LLM khi cả hai kỹ năng đều thành node đồ thị (≥ 5 tin). Đuôi hiếm để tầng embedding lo.
MIN_LLM_COUNT = 5
LLM_BATCH = 25

# Các cặp embedding hay thấy giống nhưng là kỹ năng khác nhau.
FORBIDDEN: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in (
        ("java", "javascript"), ("c", "c++"), ("c", "c#"), ("c++", "c#"), ("react", "react native"),
        ("sql", "nosql"), ("word", "excel"), ("word", "powerpoint"), ("excel", "powerpoint"),
        ("photoshop", "illustrator"), ("premiere", "after effects"), ("autocad", "revit"),
        ("ios", "android"), ("frontend", "backend"), ("front end", "back end"),
    )
)  # fmt: skip

_DIGITS = re.compile(r"\d+")
_LANGUAGE = re.compile(r"tiếng (\w+)")


def candidate_pairs(vectors: np.ndarray, threshold: float, block: int = 1024) -> pd.DataFrame:
    """Các cặp (i, j), i < j, có cosine ≥ `threshold`. `vectors` phải đã chuẩn hoá độ dài 1.

    Tính theo khối hàng để không phải giữ cả ma trận n × n trong bộ nhớ.
    """
    n = len(vectors)
    cols = np.arange(n)
    found = []
    for start in range(0, n, block):
        rows = np.arange(start, min(start + block, n))
        sims = vectors[rows] @ vectors.T
        mask = (sims >= threshold) & (cols[None, :] > rows[:, None])
        r, c = np.nonzero(mask)
        found.append(pd.DataFrame({"i": rows[r], "j": c, "score": sims[r, c].astype(float)}))
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["i", "j", "score"])


def _conflict(a: str, b: str) -> bool:
    if frozenset((a, b)) in FORBIDDEN:
        return True
    digits_a, digits_b = set(_DIGITS.findall(a)), set(_DIGITS.findall(b))
    if digits_a and digits_b and digits_a != digits_b:
        return True
    lang_a, lang_b = set(_LANGUAGE.findall(a)), set(_LANGUAGE.findall(b))
    return bool(lang_a and lang_b and lang_a != lang_b)


def merge_clusters(
    skills: list[str], counts: list[int], pairs: pd.DataFrame, max_cluster: int = MAX_CLUSTER
) -> dict[str, str]:
    """Gộp các cặp thành cụm, trả `skill → đại diện`. Đại diện là thành viên xuất hiện nhiều nhất."""
    parent = list(range(len(skills)))
    members = {i: [i] for i in range(len(skills))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ordered = pairs.sort_values(["score", "i", "j"], ascending=[False, True, True])
    for i, j in zip(ordered["i"].astype(int), ordered["j"].astype(int), strict=True):
        ri, rj = find(i), find(j)
        if ri == rj or len(members[ri]) + len(members[rj]) > max_cluster:
            continue
        if any(_conflict(skills[a], skills[b]) for a in members[ri] for b in members[rj]):
            continue
        parent[rj] = ri
        members[ri].extend(members.pop(rj))

    rep = {}
    for group in members.values():
        best = min(group, key=lambda k: (-counts[k], len(skills[k]), skills[k]))
        rep.update({skills[k]: skills[best] for k in group})
    return rep


def resolve_tiers(
    skills: pd.DataFrame,
    vectors: np.ndarray,
    auto_threshold: float = AUTO_THRESHOLD,
    llm_low: float = LLM_LOW,
    judge: Callable[[list[tuple[str, str]]], list[bool]] | None = None,
    min_llm_count: int = MIN_LLM_COUNT,
    max_cluster: int = MAX_CLUSTER,
) -> pd.DataFrame:
    """Gộp tên theo tầng embedding và (nếu có `judge`) tầng LLM.

    `skills` có cột skill, count, kind, cùng thứ tự với `vectors`. Chỉ gộp kỹ năng cùng loại.
    Trả các dòng bị gộp: skill, canonical, tier ("embedding" hoặc "llm"), score (cosine với đại diện).
    """
    names = skills["skill"].tolist()
    counts = skills["count"].to_numpy()
    kinds = skills["kind"].to_numpy()
    pairs = candidate_pairs(vectors, min(llm_low, auto_threshold) if judge else auto_threshold)
    pairs = pairs[kinds[pairs["i"].astype(int)] == kinds[pairs["j"].astype(int)]]
    auto = pairs[pairs["score"] >= auto_threshold]
    emb_rep = merge_clusters(names, counts.tolist(), auto, max_cluster)

    accepted = auto
    if judge is not None:
        band = pairs[(pairs["score"] < auto_threshold) & (pairs["score"] >= llm_low)]
        i, j = band["i"].astype(int).to_numpy(), band["j"].astype(int).to_numpy()
        band = band[np.minimum(counts[i], counts[j]) >= min_llm_count]
        index = {name: k for k, name in enumerate(names)}
        asks: dict[frozenset[str], tuple[str, str, float]] = {}
        for a, b, score in zip(band["i"].astype(int), band["j"].astype(int), band["score"], strict=True):
            ra, rb = emb_rep[names[a]], emb_rep[names[b]]
            key = frozenset((ra, rb))
            if ra != rb and not _conflict(ra, rb) and score > asks.get(key, ("", "", -1.0))[2]:
                asks[key] = (ra, rb, float(score))
        questions = list(asks.values())
        verdicts = judge([(a, b) for a, b, _ in questions])
        approved = pd.DataFrame(
            [(index[a], index[b], s) for (a, b, s), ok in zip(questions, verdicts, strict=True) if ok],
            columns=["i", "j", "score"],
        )
        accepted = pd.concat([auto, approved], ignore_index=True)

    final = merge_clusters(names, counts.tolist(), accepted, max_cluster)
    index = {name: k for k, name in enumerate(names)}
    rows = [
        (s, r, "embedding" if emb_rep[s] == r else "llm", float(vectors[index[s]] @ vectors[index[r]]))
        for s, r in final.items()
        if s != r
    ]
    merges = pd.DataFrame(rows, columns=["skill", "canonical", "tier", "score"])
    return merges.sort_values(["canonical", "skill"]).reset_index(drop=True)


def embedding_tier(
    skills: pd.DataFrame,
    vectors: np.ndarray,
    threshold: float = AUTO_THRESHOLD,
    max_cluster: int = MAX_CLUSTER,
) -> pd.DataFrame:
    """Chỉ tầng embedding: gộp tự động kỹ năng cùng loại có cosine ≥ `threshold`."""
    return resolve_tiers(skills, vectors, auto_threshold=threshold, judge=None, max_cluster=max_cluster)


JUDGE_SYSTEM = (
    "Bạn là chuyên viên tuyển dụng ở Việt Nam. Bạn chuẩn hoá tên kỹ năng trong tin tuyển dụng. "
    "Chỉ trả về JSON."
)
JUDGE_RULES = """Với mỗi cặp tên kỹ năng dưới đây, cho biết hai tên có phải CÙNG MỘT kỹ năng không.
"Cùng" nghĩa là thay được cho nhau trong yêu cầu tuyển dụng: người có kỹ năng A đáp ứng tin yêu cầu B,
và ngược lại. Viết tắt, khác chính tả, khác ngôn ngữ (Anh/Việt) của cùng một thứ là "cùng".
Một bên rộng hơn hẳn bên kia (ví dụ "chỉnh sửa ảnh video" và "chỉnh sửa video"), hoặc hai công cụ,
ngôn ngữ, chứng chỉ khác nhau (ví dụ "java" và "javascript") là "khác".

Trả về JSON dạng {"answers": [{"id": 1, "same": true}, {"id": 2, "same": false}, ...]}.

"""


def read_judgements(text: str, n: int) -> tuple[list[bool], bool]:
    """Đọc câu trả lời JSON của LLM. Trả (kết quả, đọc trọn vẹn không).

    Cặp thiếu hoặc JSON hỏng thì coi là "khác", để không gộp nhầm.
    """
    result = [False] * n
    answered = set()
    try:
        data = json.loads(extract_json(text))
        answers = data["answers"] if isinstance(data, dict) else data
        for item in answers:
            k = int(item["id"])
            if 1 <= k <= n:
                result[k - 1] = bool(item["same"])
                answered.add(k)
    except (ValueError, KeyError, TypeError):
        return [False] * n, False
    return result, len(answered) == n


def parse_judgements(text: str, n: int) -> list[bool]:
    return read_judgements(text, n)[0]


def judge_prompt(batch: list[tuple[str, str]]) -> str:
    return JUDGE_RULES + "\n".join(f"{k}. {a} | {b}" for k, (a, b) in enumerate(batch, start=1))


def judge_pairs(client, pairs: list[tuple[str, str]], batch_size: int = LLM_BATCH) -> list[bool]:
    """Hỏi LLM từng lô `batch_size` cặp. `client` là một `LLMClient` (có cache)."""
    verdicts: list[bool] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        text = client.complete(judge_prompt(batch), system=JUDGE_SYSTEM, json_output=True)
        verdicts.extend(parse_judgements(text, len(batch)))
    return verdicts
