import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from webbot.tokenizer import UnigramTokenizer
from webbot.tokenizer.unigram import (
    _lattice_params,
    e_step,
    forward_backward,
    m_step,
    prune,
    required_chars,
    viterbi,
)

JA_CORPUS = [
    "むかしむかし、ある国に王様がいました。王様は毎晩、物語を聞きました。",
    "王様は言いました。「今夜も話をしておくれ。」",
    "ドラゴンは王様の城にやってきました。すごーい、と子どもたちは言いました。",
]


def _segmentations(word, pieces):
    """Every segmentation of word as [(segment, log prob)], with the same
    byte-fallback rule as the lattice: a char that is not a piece."""
    _, unk = _lattice_params(pieces)
    if not word:
        yield []
        return
    for k in range(1, len(word) + 1):
        sub = word[:k]
        if sub in pieces:
            lp = pieces[sub]
        elif k == 1:
            lp = unk
        else:
            continue
        for rest in _segmentations(word[k:], pieces):
            yield [(sub, lp)] + rest


def _random_pieces(rng, alphabet="abcあい"):
    """Random pieces with log probabilities that sum to 1."""
    weights = {ch: rng.uniform(0.05, 1) for ch in alphabet}
    for _ in range(15):
        p = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 4)))
        weights[p] = rng.uniform(0.01, 1)
    total = sum(weights.values())
    return {p: math.log(w / total) for p, w in weights.items()}


def test_viterbi_finds_the_best_segmentation():
    rng = random.Random(0)
    for _ in range(200):
        pieces = _random_pieces(rng)
        # "z" is not a piece: byte fallback
        word = "".join(rng.choice("abcあいz") for _ in range(rng.randint(1, 9)))
        score, segs = viterbi(word, pieces)
        best = max(sum(lp for _, lp in s) for s in _segmentations(word, pieces))
        assert score == pytest.approx(best)
        assert "".join(segs) == word
        _, unk = _lattice_params(pieces)
        assert sum(pieces.get(s, unk) for s in segs) == pytest.approx(score)


def test_viterbi_exclude_forbids_that_piece():
    pieces = {"a": -3.0, "b": -3.0, "ab": -0.1}
    assert viterbi("ab", pieces)[1] == ["ab"]
    assert viterbi("ab", pieces, exclude="ab")[1] == ["a", "b"]


def test_forward_backward_matches_enumeration():
    rng = random.Random(1)
    for _ in range(100):
        pieces = _random_pieces(rng)
        word = "".join(rng.choice("abcあいz") for _ in range(rng.randint(1, 8)))
        log_z, counts = forward_backward(word, pieces)

        segs = list(_segmentations(word, pieces))
        totals = [sum(lp for _, lp in s) for s in segs]
        ref_log_z = math.log(sum(math.exp(t) for t in totals))
        ref_counts = Counter()
        for s, t in zip(segs, totals):
            for seg, _ in s:
                if seg in pieces:
                    ref_counts[seg] += math.exp(t - ref_log_z)

        assert log_z == pytest.approx(ref_log_z)
        assert set(counts) == set(ref_counts)
        for p, c in ref_counts.items():
            assert counts[p] == pytest.approx(c)


def test_em_never_decreases_likelihood():
    rng = random.Random(2)
    words = [
        ("".join(rng.choice("abcあい") for _ in range(rng.randint(1, 7))), rng.randint(1, 20))
        for _ in range(60)
    ]
    pieces = _random_pieces(rng)
    prev = -math.inf
    for _ in range(8):
        counts, loglik = e_step(words, pieces)
        assert loglik >= prev - 1e-9
        prev = loglik
        pieces = m_step(counts, required="abcあい", min_count=0)
    # It actually moved: the likelihood improved over the random start.
    assert prev > e_step(words, _random_pieces(random.Random(2)))[1] + 1


def test_m_step_is_normalized_and_keeps_required_chars():
    counts = Counter({"ab": 3.0, "a": 1.0, "rare": 0.1})
    pieces = m_step(counts, required=["a", "b"], min_count=0.5)
    assert set(pieces) == {"ab", "a", "b"}  # "rare" dropped, "b" kept
    assert sum(math.exp(lp) for lp in pieces.values()) == pytest.approx(1.0)
    assert pieces["ab"] == pytest.approx(math.log(3.0 / 4.5))


def test_prune_drops_unused_pieces_before_useful_ones():
    words = [("abcd", 100), ("ab", 1)]
    pieces = {ch: -3.0 for ch in "abcd"}
    pieces.update({"abcd": -0.5, "ab": -1.5, "bc": -1.0, "cd": -2.0})
    # Viterbi uses "abcd" for the frequent word and "ab" for the rare one;
    # "bc" and "cd" are never used, so they go first.
    assert set(prune(words, pieces, size=6)) == set("abcd") | {"abcd", "ab"}
    assert set(prune(words, pieces, size=5)) == set("abcd") | {"abcd"}


def test_required_chars_reach_coverage():
    freqs = Counter({"ああい": 10, "う": 1})  # あ 20, い 10, う 1
    assert required_chars(freqs, 0.6) == ["あ"]
    assert required_chars(freqs, 0.9) == ["あ", "い"]
    assert required_chars(freqs, 1.0) == ["あ", "い", "う"]


def _trained(vocab_size=400, corpus=JA_CORPUS * 3, **kw):
    tok = UnigramTokenizer()
    tok.train(corpus, vocab_size=vocab_size, **kw)
    return tok


def test_vocab_size_and_id_layout():
    tok = _trained(vocab_size=320)
    assert len(tok) == 320
    assert sorted(tok.vocab.values()) == list(range(256, 256 + len(tok.vocab)))
    first_special = 256 + len(tok.vocab)
    assert tok.special_tokens == {
        t: first_special + i for i, t in enumerate(tok.SPECIAL_TOKENS)
    }
    # Ids follow decreasing probability.
    lps = [tok.pieces[tok.inverse_vocab[i]] for i in range(256, first_special)]
    assert lps == sorted(lps, reverse=True)


def test_frequent_word_becomes_one_piece():
    tok = _trained()
    assert tok.encode("王様") == [tok.vocab["王様"]]


def test_roundtrip_with_byte_fallback():
    tok = _trained()
    for text in JA_CORPUS + ["見たことのない漢字「鬱」と絵文字🐍、English 123\n\n"]:
        assert tok.decode(tok.encode(text)) == text
    # Characters never seen in training are encoded as their UTF-8 bytes.
    assert tok.encode("🐍") == list("🐍".encode("utf-8"))


def test_rare_chars_below_coverage_use_bytes():
    corpus = ["ああああいいいい"] * 50 + ["鬱"]
    tok = _trained(vocab_size=280, corpus=corpus, character_coverage=0.99)
    assert "鬱" not in tok.pieces
    assert tok.encode("鬱") == list("鬱".encode("utf-8"))
    assert tok.decode(tok.encode("あ鬱い")) == "あ鬱い"


def test_vocab_is_smaller_when_the_corpus_runs_out_of_pieces():
    tok = _trained(vocab_size=5000)
    assert len(tok) < 5000
    assert len(tok) == 256 + len(tok.pieces) + len(tok.SPECIAL_TOKENS)


def test_vocab_too_small_for_required_chars_is_rejected():
    with pytest.raises(ValueError):
        _trained(vocab_size=256 + 8 + 3, character_coverage=1.0)


def test_encode_chunks_matches_encode():
    tok = _trained()
    text = "".join(JA_CORPUS)
    for size in (1, 5, 50):
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        assert list(tok.encode_chunks(chunks)) == tok.encode(text)


def test_special_tokens(tmp_path):
    tok = _trained()
    qa = "<|question|>王様は？<|answer|>城<|endoftext|>"
    ids = tok.encode(qa, allowed_special="all")
    assert ids[0] == tok.special_tokens["<|question|>"]
    assert ids[-1] == tok.special_tokens[tok.EOT_TOKEN]
    assert tok.decode(ids) == qa
    assert not set(tok.encode(qa)) & set(tok.special_tokens.values())

    tok.save(tmp_path / "uni.json")
    loaded = UnigramTokenizer.load(tmp_path / "uni.json")
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.pieces == tok.pieces
    assert loaded.encode(qa, allowed_special="all") == ids


def test_parallel_counting_gives_same_tokenizer():
    single = _trained()
    parallel = _trained(num_workers=2, piece_chars=16)
    assert parallel.pieces == single.pieces


def test_required_chars_survive_a_tight_budget():
    # 50 two-letter katakana words are each covered by their own piece, so
    # EM leaves the single characters with tiny probabilities, below those
    # of the word pieces. The final cut to the budget (from up to 1.1x the
    # budget) goes by probability, yet every required character must stay
    # a piece rather than fall back to bytes.
    import itertools

    kana = "アイウエオカキクケコ"
    words = ["".join(p) for p in itertools.permutations(kana, 2)][:50]
    corpus = [w for w in words for _ in range(10)] + ["ぬいぐるみ"] * 100
    required = required_chars(Counter(corpus), 1.0)
    budget = len(required) + 20
    tok = _trained(vocab_size=256 + 8 + budget, corpus=corpus, character_coverage=1.0)
    assert len(tok.pieces) == budget
    assert set(required) <= set(tok.pieces)
