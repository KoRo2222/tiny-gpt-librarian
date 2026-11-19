import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.retrieval import BM25Index, char_bigrams, normalize, token_analyzer
from webbot.tokenizer import BPETokenizer


def test_normalize_and_bigrams():
    assert normalize("ＡＢＣ ｶﾀｶﾅ") == "abc カタカナ"
    assert char_bigrams("東京都 の\tAI") == ["東京", "京都", "の", "ai"]
    assert char_bigrams("") == []


def _reference_scores(docs, query, analyzer, k1, b):
    """BM25 by the formula, one document and one term at a time."""
    analyzed = [analyzer(d) for d in docs]
    n = len(docs)
    avgdl = sum(map(len, analyzed)) / n
    out = []
    for terms in analyzed:
        s = 0.0
        for t in dict.fromkeys(analyzer(query)):
            tf = terms.count(t)
            if tf == 0:
                continue
            df = sum(t in other for other in analyzed)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(terms) / avgdl))
        out.append(s)
    return out


def _random_corpus(rng, n_docs=30):
    alphabet = "あいうえお山川"
    return ["".join(rng.choice(alphabet + " ") for _ in range(rng.randint(1, 40))) for _ in range(n_docs)]


def test_scores_match_the_formula():
    rng = random.Random(0)
    for trial in range(20):
        docs = _random_corpus(rng)
        k1, b = rng.uniform(0.5, 2.0), rng.uniform(0.0, 1.0)
        index = BM25Index(k1=k1, b=b).build(docs)
        for _ in range(5):
            query = "".join(rng.choice("あいうえお山川 ") for _ in range(rng.randint(1, 8)))
            want = _reference_scores(docs, query, char_bigrams, k1, b)
            assert index.scores(query) == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_rare_terms_weigh_more():
    docs = ["東京は都市", "東京の天気", "東京と京都", "富士山の天気"]
    index = BM25Index().build(docs)
    # "富士" occurs in one document, "東京" in three.
    assert index.search("富士", 1)[0][0] == 3
    s = index.scores("東京富士")
    assert s[3] > max(s[:3])


def test_shorter_documents_score_higher_unless_b_is_zero():
    docs = ["山川", "山川あいうえおかきくけこ", "かきく"]
    s = BM25Index(b=0.75).build(docs).scores("山川")
    assert s[0] > s[1] > 0
    s0 = BM25Index(b=0.0).build(docs).scores("山川")
    assert s0[0] == pytest.approx(s0[1])


def test_repeated_terms_saturate():
    docs = ["山川" + " 山川" * n + " あい" * (5 - n) for n in range(5)] + ["かき"] * 5
    s = BM25Index(k1=1.2, b=0.0).build(docs).scores("山川")
    gains = [s[i + 1] - s[i] for i in range(4)]
    assert all(g > 0 for g in gains)
    assert all(a > b for a, b in zip(gains, gains[1:]))  # diminishing
    idf = math.log(1 + (10 - 5 + 0.5) / (5 + 0.5))
    assert s[4] < idf * (1.2 + 1)  # bounded by idf * (k1 + 1)


def test_search_is_the_sorted_top_k_with_ties_to_the_lower_id():
    rng = random.Random(1)
    for _ in range(30):
        docs = _random_corpus(rng, n_docs=rng.randint(1, 40))
        docs += docs[: len(docs) // 2]  # duplicates: exact ties
        index = BM25Index().build(docs)
        query = "".join(rng.choice("あいうえお山川") for _ in range(4))
        s = index.scores(query)
        ranked = sorted((d for d in range(len(docs)) if s[d] > 0), key=lambda d: (-s[d], d))
        for k in (1, 3, 10, 100):
            assert [d for d, _ in index.search(query, k)] == ranked[:k]


def test_unknown_or_empty_query_finds_nothing():
    index = BM25Index().build(["山川", "あいう"])
    assert index.search("ゼロ") == []
    assert index.search("") == []
    assert BM25Index().build([]).search("山") == []


def test_idf_stays_positive_when_every_document_matches():
    index = BM25Index().build(["山川"] * 4)
    assert index.idf(4) > 0
    assert index.search("山川", 4)[0][1] > 0


def test_save_and_load(tmp_path):
    docs = ["東京は都市", "東京の天気", "京都と奈良", "富士山の天気"]
    index = BM25Index(k1=1.5, b=0.5).build(docs)
    index.save(tmp_path / "idx")
    loaded = BM25Index.load(tmp_path / "idx")
    assert (loaded.k1, loaded.b) == (1.5, 0.5)
    for q in ("東京", "天気の京都", "奈良"):
        assert loaded.search(q, 4) == index.search(q, 4)


def test_token_analyzer(tmp_path):
    tok = BPETokenizer()
    tok.train(["東京の天気は晴れ。京都の天気は雨。"] * 20, vocab_size=300)
    analyzer = token_analyzer(tok)
    docs = ["東京の天気", "京都の天気", "晴れと雨"]
    index = BM25Index(analyzer).build(docs)
    assert index.search("東京", 1)[0][0] == 0
    index.save(tmp_path / "idx")
    assert BM25Index.load(tmp_path / "idx", analyzer).search("京都", 3) == index.search("京都", 3)
    assert index.scores("京都の天気") == pytest.approx(
        _reference_scores(docs, "京都の天気", analyzer, 1.2, 0.75), rel=1e-12)
