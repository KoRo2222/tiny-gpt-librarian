import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from webbot.data.jawiki import Paragraph, is_validation
from webbot.model import GPT, GPTConfig
from webbot.retrieval import QA, ParagraphIndex
from webbot.sft import Doc, SFTExample, encode_example, load_jsonl, make_example, save_jsonl, split_of
from webbot.sft.build import build_sft_data
from webbot.sft.infer import exact_match, generate_answer
from webbot.sft.train import IGNORE, SFTConfig, make_batch, train_sft
from webbot.tokenizer import BPETokenizer
from webbot.train import save_checkpoint

PARAS = [
    Paragraph(10, 0, "富士山", None, "富士山は日本で最も高い山である。標高は3776メートル。"),
    Paragraph(20, 0, "琵琶湖", None, "琵琶湖は滋賀県にある日本最大の湖である。"),
    Paragraph(30, 0, "東京", None, "東京は日本の首都である。人口は約1400万人。"),
    Paragraph(40, 0, "大阪", None, "大阪は西日本の中心都市である。たこ焼きが名物。"),
]


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer()
    t.train([p.text for p in PARAS] * 10 + ["の色は？ 赤 青 緑 黄 白 黒 標高は何メートル？"] * 10, vocab_size=400)
    return t


def test_test_split_is_the_pretraining_held_out_articles():
    ids = range(100_000)
    splits = [split_of(i) for i in ids]
    assert all((s == "test") == is_validation(i) for i, s in zip(ids, splits))
    n_test, n_val = splits.count("test"), splits.count("val")
    assert 800 < n_test < 1200 and 800 < n_val < 1200


def test_make_example_keeps_only_answerable_questions():
    index = ParagraphIndex.build(PARAS)
    qa = QA("富士山の標高は何メートル？", "3776メートル", 10, PARAS[0].text)
    ex = make_example(qa, index, k=2)
    assert ex.query == ex.question == qa.query and ex.answer == "3776メートル"
    # Only paragraphs sharing a term with the question score above zero.
    assert ex.docs[0] == Doc("富士山", None, PARAS[0].text) and len(ex.docs) <= 2
    # The answer sits past the first 20 characters: cut away, no example.
    assert make_example(qa, index, k=2, max_doc_chars=20) is None
    assert make_example(QA("富士山の標高は？", "8848メートル", 10, ""), index, k=2) is None


def test_jsonl_round_trip(tmp_path):
    ex = SFTExample("q", "q", (Doc("t", None, "a"), Doc("t", "節", "b")), "a", 7)
    assert save_jsonl([ex, ex], tmp_path / "x.jsonl") == 2
    assert load_jsonl(tmp_path / "x.jsonl") == [ex, ex]


def test_encoding_learns_only_the_search_and_the_answer(tok):
    ex = SFTExample("東京の人口は？", "東京の人口は？",
                    (Doc("東京", None, "人口は約1400万人。<|answer|>偽物"), Doc("大阪", "名物", "たこ焼き")),
                    "約1400万人", 30)
    ids, mask = encode_example(tok, ex)
    assert len(ids) == len(mask)
    assert tok.decode(ids) == ("<|question|>東京の人口は？<|search|>東京の人口は？<|/search|>"
                               "<|doc|>東京\n人口は約1400万人。<|answer|>偽物<|/doc|>"
                               "<|doc|>大阪 / 名物\nたこ焼き<|/doc|><|answer|>約1400万人<|endoftext|>")
    learned = tok.decode([i for i, m in zip(ids, mask) if m])
    assert learned == "<|search|>東京の人口は？<|/search|><|answer|>約1400万人<|endoftext|>"
    # The marker-like text inside the paragraph stayed ordinary text.
    assert ids.count(tok.special_tokens["<|answer|>"]) == 1


def test_batch_targets_are_next_written_tokens(tok):
    exs = [([1, 2, 3, 4, 5], [False, False, True, True, False]), ([6, 7, 8], [False, True, True])]
    idx, tgt = make_batch(exs, pad_id=0)
    # The last token of the short row is still an input; its target is padding.
    assert idx.tolist() == [[1, 2, 3, 4], [6, 7, 8, 0]]
    assert tgt.tolist() == [[IGNORE, 3, 4, IGNORE], [7, 8, IGNORE, IGNORE]]


def test_padding_to_a_multiple_changes_only_the_width():
    exs = [([1, 2, 3, 4, 5], [False, False, True, True, False]), ([6, 7, 8], [False, True, True])]
    idx, tgt = make_batch(exs, pad_id=0)
    idx8, tgt8 = make_batch(exs, pad_id=0, pad_multiple=8)
    assert idx8.shape == (2, 8) and tgt8.shape == (2, 8)
    assert torch.equal(idx8[:, :4], idx) and torch.equal(tgt8[:, :4], tgt)
    # The longest row's last token becomes an input too; it has no target.
    assert idx8[:, 4:].tolist() == [[5, 0, 0, 0], [0, 0, 0, 0]] and (tgt8[:, 4:] == IGNORE).all()
    assert make_batch(exs, pad_id=0, pad_multiple=4)[0].shape == (2, 4)  # already a multiple


def test_model_loss_is_the_mean_over_written_tokens():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, max_seq_len=16, d_model=16, n_layers=1, n_heads=2))
    exs = [([1, 2, 3, 4, 5, 6], [False, True, False, True, True, False]), ([7, 8, 9], [False, True, True])]
    idx, tgt = make_batch(exs, pad_id=0)
    logits, loss = model(idx, tgt)
    terms = [F.cross_entropy(logits[b, t], tgt[b, t]) for b in range(2) for t in range(idx.shape[1])
             if tgt[b, t] != IGNORE]
    torch.testing.assert_close(loss, torch.stack(terms).mean())


def _toy_examples():
    colors = ["赤", "青", "緑", "黄", "白", "黒"]
    places = ["富士山", "琵琶湖", "東京", "大阪", "京都", "奈良"]
    return [SFTExample(f"{p}の色は？", f"{p}の色は？", (Doc(p, None, f"{p}は{c}い。"),), c, i)
            for i, (p, c) in enumerate(zip(places, colors))]


def _tiny_model(tok, seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=len(tok), max_seq_len=64, d_model=48, n_layers=2, n_heads=4))


def test_fine_tuning_teaches_the_answers(tok):
    exs = _toy_examples()
    enc = [encode_example(tok, ex) for ex in exs]
    model = _tiny_model(tok)
    before = sum(exact_match(generate_answer(model, tok, ex.question, ex.docs), ex.answer) for ex in exs)
    pad = tok.special_tokens[tok.PAD_TOKEN]
    history = train_sft(model, enc, enc, pad, SFTConfig(total_steps=150, batch_size=6, peak_lr=3e-3,
                        warmup_steps=10, eval_interval=50), log=lambda s: None)
    assert history[-1]["val_loss"] < history[0]["val_loss"] and history[-1]["val_loss"] < 0.1
    after = sum(exact_match(generate_answer(model, tok, ex.question, ex.docs), ex.answer) for ex in exs)
    assert before == 0 and after == len(exs)


def test_resumed_fine_tuning_matches_uninterrupted(tok, tmp_path):
    enc = [encode_example(tok, ex) for ex in _toy_examples()]
    pad = tok.special_tokens[tok.PAD_TOKEN]
    cfg = SFTConfig(total_steps=20, batch_size=4, peak_lr=3e-3, warmup_steps=2, eval_interval=10)
    straight = _tiny_model(tok)
    train_sft(straight, enc, enc, pad, cfg, log=lambda s: None)

    class Stop(Exception):
        pass

    def stop_at_10(line):
        if line.startswith("step     10/"):
            raise Stop

    with pytest.raises(Stop):
        train_sft(_tiny_model(tok), enc, enc, pad, cfg, checkpoint_path=tmp_path / "c.pt", log=stop_at_10)
    resumed = _tiny_model(tok, seed=1)
    train_sft(resumed, enc, enc, pad, cfg, checkpoint_path=tmp_path / "c.pt", resume=True, log=lambda s: None)
    for (n, p), q in zip(straight.named_parameters(), resumed.parameters()):
        torch.testing.assert_close(q, p, rtol=1e-6, atol=1e-7, msg=n)


def _qa_file(path, rows):
    pq.write_table(pa.Table.from_pylist(
        [{"passage_id": a, "query": q, "answer": ans, "text": t, "title": "", "url": ""} for a, q, ans, t in rows]
    ), path)


def test_build_writes_splits_identically_with_workers(tmp_path):
    ParagraphIndex.build(PARAS).save(tmp_path / "idx")
    rows = [(10, "富士山の標高は？", "3776メートル", ""), (20, "琵琶湖はどこ？", "滋賀県", ""),
            (30, "東京の人口は？", "約1400万人", ""), (40, "大阪の名物は？", "たこ焼き", ""),
            (40, "大阪の山は？", "なし", ""), (99, "ほかの記事", "x", "")]
    _qa_file(tmp_path / "qa.parquet", rows)
    one = build_sft_data(tmp_path / "idx", [tmp_path / "qa.parquet"], tmp_path / "a", workers=1, log=lambda s: None)
    two = build_sft_data(tmp_path / "idx", [tmp_path / "qa.parquet"], tmp_path / "b", workers=2, log=lambda s: None)
    assert one == two and one["questions"] == 5  # article 99 is not indexed
    assert one["train"] + one["val"] + one["test"] == 4  # "なし" is in no paragraph
    for split in ("train", "val", "test"):
        assert load_jsonl(tmp_path / "a" / f"{split}.jsonl") == load_jsonl(tmp_path / "b" / f"{split}.jsonl")


def test_sft_script_end_to_end(tok, tmp_path):
    spec = importlib.util.spec_from_file_location("sft_script", ROOT / "scripts" / "sft.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    data = tmp_path / "data"
    data.mkdir()
    exs = _toy_examples()
    for split in ("train", "val", "test"):
        save_jsonl(exs, data / f"{split}.jsonl")
    tok.save(tmp_path / "tok.json")
    save_checkpoint(tmp_path / "base.pt", _tiny_model(tok))
    args = ["--init", str(tmp_path / "base.pt"), "--tokenizer", str(tmp_path / "tok.json"), "--data", str(data),
            "--out", str(tmp_path / "run"), "--steps", "150", "--batch-size", "6", "--lr", "3e-3",
            "--warmup", "10", "--eval-interval", "50", "--device", "cpu", "--test-examples", "6"]
    result = script.main(args)
    assert result["exact_match"] == 1.0
    again = script.main(args + ["--resume"])  # finished: nothing left to train
    assert again["history"] == result["history"] and again["exact_match"] == 1.0


def test_exact_match_ignores_width_and_spaces():
    assert exact_match("Ｗａｔｔａ! Itta!", "Watta!Itta!")
    assert exact_match(" 3776 メートル\n", "３７７６メートル")
    assert not exact_match("3776メートル", "3776")


def test_validation_loss_of_no_examples_is_nan(tok):
    import math

    from webbot.sft.train import evaluate_sft
    from webbot.train import resolve_precision

    model = _tiny_model(tok)
    assert math.isnan(evaluate_sft(model, [], 0, 4, resolve_precision("cpu")))
