import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data.jawiki import Paragraph
from webbot.model import GPT, GPTConfig
from webbot.retrieval import QA, ParagraphIndex
from webbot.sft import SFTExample, encode_example, make_example
from webbot.sft.agent import ask, evaluate_end_to_end, write_query
from webbot.sft.train import SFTConfig, train_sft
from webbot.tokenizer import BPETokenizer

COLORS = ["赤", "青", "緑", "黄", "白", "黒"]
PLACES = ["富士", "琵琶", "東京", "大阪", "京都", "奈良"]
PARAS = [Paragraph(i, 0, p, None, f"{p}の色は{c}い。{p}は有名だ。") for i, (p, c) in enumerate(zip(PLACES, COLORS))]
QAS = [QA(f"{p}の色は？", c, i, PARAS[i].text) for i, (p, c) in enumerate(zip(PLACES, COLORS))]


@pytest.fixture(scope="module")
def setup():
    tok = BPETokenizer()
    tok.train([p.text for p in PARAS] * 20 + [qa.query for qa in QAS] * 20, vocab_size=400)
    index = ParagraphIndex.build(PARAS)
    examples = [make_example(qa, index, k=2) for qa in QAS]
    assert all(examples)
    return tok, index, examples


def _model(tok, seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=len(tok), max_seq_len=128, d_model=48, n_layers=2, n_heads=4))


def test_fine_tuned_model_searches_and_answers(setup):
    tok, index, examples = setup
    model = _model(tok)
    before = evaluate_end_to_end(model, tok, index, QAS)
    enc = [encode_example(tok, ex) for ex in examples]
    train_sft(model, enc, enc, tok.special_tokens[tok.PAD_TOKEN],
              SFTConfig(total_steps=200, batch_size=6, peak_lr=3e-3, warmup_steps=10, eval_interval=100),
              log=lambda s: None)
    for qa in QAS:
        reply = ask(model, tok, index, qa.query, k=2)
        assert reply.query == qa.query  # it learned to search with the question
        assert reply.docs[0].title == PARAS[qa.article_id].title
        assert reply.answer == qa.answer
    after = evaluate_end_to_end(model, tok, index, QAS, k=2)
    assert after["exact_match"] == 1.0 and after["query_is_question"] == 1.0 and after["answer_in_docs"] == 1.0
    assert before["exact_match"] < 0.5
    # A question whose answer is in no paragraph counts against answer_in_docs.
    unanswerable = QA("富士の高さは？", "3776メートル", 0, "")
    mixed = evaluate_end_to_end(model, tok, index, [QAS[0], unanswerable], k=2)
    assert mixed["answer_in_docs"] == 0.5 and mixed["exact_match"] == 0.5


def test_no_query_falls_back_to_the_question(setup, monkeypatch):
    tok, index, _ = setup
    import webbot.sft.agent as agent

    monkeypatch.setattr(agent, "write_query", lambda *a, **k: "")
    reply = ask(_model(tok), tok, index, QAS[2].query, k=2)
    assert reply.query == QAS[2].query and reply.docs[0].title == "東京"
    # Searching with the question also when asked not to use the model's query.
    monkeypatch.setattr(agent, "write_query", lambda *a, **k: "奈良")
    assert ask(_model(tok), tok, index, QAS[2].query, k=2, use_model_query=False).query == QAS[2].query


def test_query_stops_at_any_special_token(setup, monkeypatch):
    tok, _, _ = setup
    model = _model(tok)
    text = tok.encode("東京の色")
    answer = tok.special_tokens["<|answer|>"]

    def fake_generate(idx, max_new_tokens, **kw):
        return torch.cat([idx, torch.tensor([text + [answer] + tok.encode("赤")])], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    assert write_query(model, tok, "東京の色は？") == "東京の色"


def test_ask_script_prints_query_paragraphs_and_answer(setup, tmp_path, capsys, monkeypatch):
    import importlib.util

    import webbot.sft.agent as agent
    from webbot.train import save_checkpoint

    tok, index, _ = setup
    monkeypatch.setattr(agent, "write_query", lambda *a, **k: "")  # search with the question
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("ask_script", root / "scripts" / "ask.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    save_checkpoint(tmp_path / "m.pt", _model(tok))
    tok.save(tmp_path / "tok.json")
    index.save(tmp_path / "idx")
    script.main(["--model", str(tmp_path / "m.pt"), "--tokenizer", str(tmp_path / "tok.json"),
                 "--index", str(tmp_path / "idx"), "--k", "2", "東京の色は？"])
    out = capsys.readouterr().out
    assert "検索語: 東京の色は？" in out and "段落: 東京: 東京の色は" in out and out.count("答え: ") == 1
    # A query that finds nothing says so.
    monkeypatch.setattr(agent, "write_query", lambda *a, **k: "ゼロ")
    script.main(["--model", str(tmp_path / "m.pt"), "--tokenizer", str(tmp_path / "tok.json"),
                 "--index", str(tmp_path / "idx"), "--k", "2", "東京の色は？"])
    assert "段落: 見つかりませんでした" in capsys.readouterr().out
