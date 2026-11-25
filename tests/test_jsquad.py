import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import webbot.eval.jsquad as jsquad
from webbot.eval import JSQuADItem, char_f1, evaluate_jsquad, load_jsquad, score
from webbot.sft.data import Doc


def _reference_f1(pred, ref):
    """Character F1 by counting one character at a time."""
    pred, ref = list(pred), list(ref)
    remaining = list(ref)
    common = 0
    for ch in pred:
        if ch in remaining:
            remaining.remove(ch)
            common += 1
    if not common:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


def test_char_f1_matches_counting():
    for pred, ref in [("小笠原諸島", "小笠原諸島を除く日本"), ("ああい", "あいい"), ("東京", "大阪"), ("abc", "abc")]:
        assert char_f1(pred, ref) == pytest.approx(_reference_f1(pred, ref))
    assert char_f1("小笠原諸島", "小笠原諸島を除く日本") == pytest.approx(2 / 3)  # P 1, R 1/2


def test_score_uses_the_closest_reference_after_normalizing():
    s = score("　小笠原 諸島", ["小笠原諸島を除く日本", "小笠原諸島"])
    assert s == {"exact_match": 1.0, "f1": 1.0, "contains_answer": 1.0}
    s = score("北海道と小笠原諸島", ["小笠原諸島"])
    assert s["exact_match"] == 0.0 and s["contains_answer"] == 1.0
    assert s["f1"] == pytest.approx(char_f1("北海道と小笠原諸島", "小笠原諸島"))
    assert score("雨季", ["小笠原諸島"]) == {"exact_match": 0.0, "f1": 0.0, "contains_answer": 0.0}


def test_load_splits_title_and_dedups_answers(tmp_path):
    data = {"data": [{"title": "梅雨", "paragraphs": [{
        "context": "梅雨 [SEP] 梅雨は東アジアの気象現象。",
        "qas": [{"id": "q1", "question": "どこの現象？", "is_impossible": False,
                 "answers": [{"text": "東アジア", "answer_start": 6}, {"text": "東アジア", "answer_start": 6},
                             {"text": "東アジアの気象", "answer_start": 6}]}]}]}]}
    path = tmp_path / "d.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert load_jsquad(path) == [JSQuADItem("q1", "どこの現象？", "梅雨", "梅雨は東アジアの気象現象。",
                                            ("東アジア", "東アジアの気象"))]


def test_evaluation_gives_the_paragraph_and_averages(monkeypatch):
    seen = []

    def fake_answer(model, tok, question, docs, max_new_tokens=48):
        seen.append((question, docs))
        return {"q1": "東アジア", "q2": "北海道"}[question]

    monkeypatch.setattr(jsquad, "generate_answer", fake_answer)
    items = [JSQuADItem("1", "q1", "梅雨", "本文1", ("東アジア",)), JSQuADItem("2", "q2", "梅雨", "本文2", ("沖縄",))]
    result = evaluate_jsquad(None, None, items)
    assert seen == [("q1", (Doc("梅雨", None, "本文1"),)), ("q2", (Doc("梅雨", None, "本文2"),))]
    assert result == {"exact_match": 0.5, "f1": 0.5, "contains_answer": 0.5}


def test_jsquad_questions_become_sft_examples():
    from webbot.eval import to_sft_examples
    from webbot.sft import SFTExample

    items = [JSQuADItem("1", "どこの現象？", "梅雨", "梅雨は東アジアの気象現象。", ("東アジア", "東アジアの気象"))]
    assert to_sft_examples(items) == [
        SFTExample("どこの現象？", "どこの現象？", (Doc("梅雨", None, "梅雨は東アジアの気象現象。"),), "東アジア", -1)
    ]


def test_add_jsquad_extends_only_the_training_split(tmp_path):
    import importlib.util

    from webbot.sft import SFTExample, load_jsonl, save_jsonl

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("add_jsquad", root / "scripts" / "add_jsquad.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sft = tmp_path / "sft"
    sft.mkdir()
    ex = SFTExample("q", "q", (Doc("t", None, "a"),), "a", 7)
    for split, n in (("train", 3), ("val", 1), ("test", 2)):
        save_jsonl([ex] * n, sft / f"{split}.jsonl")
    data = {"data": [{"title": "梅雨", "paragraphs": [{"context": "梅雨 [SEP] 東アジアの現象。", "qas": [
        {"id": f"q{i}", "question": f"問{i}", "answers": [{"text": "東アジア", "answer_start": 0}]} for i in range(4)]}]}]}
    (tmp_path / "j.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    counts = script.main(["--sft", str(sft), "--jsquad", str(tmp_path / "j.json"), "--out", str(tmp_path / "out")])
    assert counts == {"train_base": 3, "train_jsquad": 4, "train": 7}
    train = load_jsonl(tmp_path / "out" / "train.jsonl")
    assert train[:3] == [ex] * 3 and [e.question for e in train[3:]] == ["問0", "問1", "問2", "問3"]
    for split in ("val", "test"):
        assert load_jsonl(tmp_path / "out" / f"{split}.jsonl") == load_jsonl(sft / f"{split}.jsonl")
