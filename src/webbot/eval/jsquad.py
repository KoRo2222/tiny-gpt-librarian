"""Reading comprehension on JSQuAD (JGLUE; Japanese SQuAD-style, questions
written by people about Wikipedia paragraphs).

Each question comes with its paragraph ("title [SEP] text"), which is
given to the model as the single <|doc|>; the model answers after
<|answer|>. Scores are those of SQuAD, at the character level as usual for
Japanese: exact match and character F1 against the closest of the
reference answers, after normalize_answer (NFKC, whitespace removed).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from webbot.sft.data import Doc
from webbot.sft.infer import generate_answer, normalize_answer

SEP = " [SEP] "


@dataclass(frozen=True)
class JSQuADItem:
    id: str
    question: str
    title: str
    text: str
    answers: tuple[str, ...]


def load_jsquad(path: str | Path) -> list[JSQuADItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    items = []
    for article in data:
        for paragraph in article["paragraphs"]:
            title, sep, text = paragraph["context"].partition(SEP)
            if not sep:  # no title marker
                title, text = article.get("title", ""), paragraph["context"]
            for qa in paragraph["qas"]:
                answers = tuple(dict.fromkeys(a["text"] for a in qa["answers"]))
                items.append(JSQuADItem(qa["id"], qa["question"], title, text, answers))
    return items


def char_f1(prediction: str, reference: str) -> float:
    """F1 of the normalized characters (as multisets)."""
    pred, ref = Counter(normalize_answer(prediction)), Counter(normalize_answer(reference))
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def score(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    """exact match, F1 and containment against the best reference."""
    pred = normalize_answer(prediction)
    return {
        "exact_match": float(any(pred == normalize_answer(a) for a in answers)),
        "f1": max(char_f1(prediction, a) for a in answers),
        "contains_answer": float(any(normalize_answer(a) in pred for a in answers)),
    }


def evaluate_jsquad(
    model, tokenizer, items: Sequence[JSQuADItem], log: Callable[[str], None] | None = None,
    max_new_tokens: int = 48,
) -> dict[str, float]:
    """Mean scores over the items, the paragraph given as the only document."""
    totals = Counter()
    for i, item in enumerate(items):
        pred = generate_answer(model, tokenizer, item.question, (Doc(item.title, None, item.text),),
                               max_new_tokens=max_new_tokens)
        totals.update(score(pred, item.answers))
        if log is not None and i < 8:
            log(f"Q: {item.question} | gold: {' / '.join(item.answers)} | pred: {pred}")
    return {k: totals[k] / len(items) for k in ("exact_match", "f1", "contains_answer")} if items else {}


def to_sft_examples(items: Sequence[JSQuADItem]) -> list:
    """JSQuAD questions as SFT examples in the same layout as the
    auto-wiki-qa ones: the question is the search query, the question's own
    paragraph the one document (as in evaluate_jsquad), the first reference
    the answer. JSQuAD has no article ids: article_id is -1."""
    from webbot.sft.data import SFTExample

    return [SFTExample(it.question, it.question, (Doc(it.title, None, it.text),), it.answers[0], -1)
            for it in items]
