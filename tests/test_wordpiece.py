import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from webbot.tokenizer import WordPieceTokenizer
from webbot.tokenizer.pretokenize import count_pretokens, required_chars
from webbot.tokenizer.wordpiece import learn_merges, merge_symbols, merge_word, split_word

JA_CORPUS = [
    "むかしむかし、ある国に王様がいました。王様は毎晩、物語を聞きました。",
    "王様は言いました。「今夜も話をしておくれ。」",
    "ドラゴンは王様の城にやってきました。すごーい、と子どもたちは言いました。",
]


def _word_freqs(texts, coverage=1.0):
    counts = count_pretokens(texts)
    alphabet = set(required_chars(counts, coverage))
    freqs = {}
    for token, f in counts.items():
        w = tuple(split_word(token, alphabet))
        freqs[w] = freqs.get(w, 0) + f
    return freqs


def _reference_merges(word_freqs, num_merges):
    """Naive WordPiece: recount every symbol and pair before each merge."""
    words = dict(word_freqs)
    merges = []
    for _ in range(num_merges):
        sym, pairs = Counter(), Counter()
        for w, f in words.items():
            for s in w:
                if s is not None:
                    sym[s] += f
            for a, b in zip(w, w[1:]):
                if a is not None and b is not None:
                    pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p] / (sym[p[0]] * sym[p[1]]), p))
        merged = merge_symbols(*best)
        new = {}
        for w, f in words.items():
            nw = merge_word(w, best, merged)
            new[nw] = new.get(nw, 0) + f
        words = new
        merges.append(best)
    return merges


def _random_texts(seed, alphabet, n_docs=30):
    rng = random.Random(seed)
    return [
        " ".join(
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 9)))
            for _ in range(rng.randint(1, 30))
        )
        for _ in range(n_docs)
    ]


def test_incremental_training_matches_full_recount():
    corpora = [
        (_random_texts(0, "ab"), 1.0),  # heavy ties; a+##bb and ab+##b collide
        (_random_texts(1, "abcあいう"), 1.0),
        (_random_texts(2, "abcあいう鬱"), 0.95),  # rare chars become barriers
        (JA_CORPUS, 1.0),
        (["aaaaaaa aaa aaaa"], 1.0),
    ]
    for texts, coverage in corpora:
        wf = _word_freqs(texts, coverage)
        assert learn_merges(wf, 500) == _reference_merges(wf, 500)


def test_merges_by_score_not_frequency():
    # (a, ##b) occurs 10 times but "a" is everywhere; (x, ##y) occurs once,
    # and x, y never occur apart, so its score 1/(1*1) beats 10/(110*10).
    wf = {
        ((False, "a"), (True, "b")): 10,
        ((False, "a"),): 100,
        ((False, "x"), (True, "y")): 1,
    }
    assert learn_merges(wf, 1) == [((False, "x"), (True, "y"))]


def test_barrier_chars_never_merge():
    wf = {tuple(split_word("a鬱b", {"a", "b"})): 5}
    assert wf == {((False, "a"), None, (True, "b")): 5}
    assert learn_merges(wf, 10) == []


def _hand_built(tokens):
    tok = WordPieceTokenizer()
    tok._build(tokens)
    return tok


def test_encode_is_greedy_longest_match_first():
    F, T = False, True
    tok = _hand_built([(F, "a"), (F, "ab"), (T, "b"), (T, "c"), (T, "d"), (T, "cd"), (F, "b")])
    v = tok.vocab
    assert tok.encode("abcd") == [v[F, "ab"], v[T, "cd"]]
    # Word-initial and continuation pieces are different tokens.
    assert tok.encode("bb") == [v[F, "b"], v[T, "b"]]
    # No continuation piece for "a": byte fallback inside the word.
    assert tok.encode("ba") == [v[F, "b"], ord("a")]


def test_each_step_takes_the_longest_match():
    rng = random.Random(3)
    for _ in range(50):
        tokens = {(False, c) for c in "abc"} | {(True, c) for c in "abc"}
        for _ in range(12):
            text = "".join(rng.choice("abc") for _ in range(rng.randint(2, 4)))
            tokens.add((rng.random() < 0.5, text))
        tok = _hand_built(sorted(tokens))
        word = "".join(rng.choice("abcz") for _ in range(rng.randint(1, 10)))
        i = 0
        for tid in tok._encode_word(word):
            if tid < 256:  # byte fallback: no piece starts with this char
                assert word[i] == "z"
                i += 1
                continue
            cont, text = tok.tokens[tid - 256]
            assert cont == (i > 0) and word.startswith(text, i)
            assert not any(
                (i > 0, word[i:j]) in tok.vocab for j in range(i + len(text) + 1, len(word) + 1)
            )
            i += len(text)
        assert i == len(word)


def _trained(vocab_size=400, corpus=JA_CORPUS * 3, **kw):
    tok = WordPieceTokenizer()
    tok.train(corpus, vocab_size=vocab_size, **kw)
    return tok


def test_vocab_size_and_id_layout():
    tok = _trained(vocab_size=340)
    assert len(tok) == 340
    assert sorted(tok.vocab.values()) == list(range(256, 256 + len(tok.tokens)))
    first = 256 + len(tok.tokens)
    assert tok.special_tokens == {t: first + i for i, t in enumerate(tok.SPECIAL_TOKENS)}


def test_frequent_word_becomes_one_piece():
    tok = _trained()
    assert tok.encode("王様") == [tok.vocab[(False, "王様")]]


def test_roundtrip_with_hashes_and_byte_fallback():
    corpus = JA_CORPUS + ["# 見出し ## 小見出し ### a#b ##x"] * 3
    tok = _trained(corpus=corpus)
    for text in corpus + ["見たことのない漢字「鬱」と絵文字🐍、#### ###x\n"]:
        assert tok.decode(tok.encode(text)) == text
    assert tok.encode("🐍") == list("🐍".encode("utf-8"))


def test_rare_chars_below_coverage_use_bytes():
    corpus = ["ああああいいいい"] * 50 + ["あ鬱い"]
    tok = _trained(vocab_size=280, corpus=corpus, character_coverage=0.99)
    assert all("鬱" not in text for _, text in tok.tokens)
    assert tok.decode(tok.encode("あ鬱い")) == "あ鬱い"


def test_alphabet_too_large_is_rejected():
    with pytest.raises(ValueError):
        _trained(vocab_size=256 + 8 + 3)


def test_encode_chunks_matches_encode():
    tok = _trained()
    text = "".join(JA_CORPUS)
    for size in (1, 5, 50):
        chunks = [text[i : i + size] for i in range(0, len(text), size)]
        assert list(tok.encode_chunks(chunks)) == tok.encode(text)


def test_special_tokens_and_save_load(tmp_path):
    tok = _trained()
    qa = "<|question|>王様は？<|answer|>城<|endoftext|>"
    ids = tok.encode(qa, allowed_special="all")
    assert ids[0] == tok.special_tokens["<|question|>"]
    assert ids[-1] == tok.special_tokens[tok.EOT_TOKEN]
    assert tok.decode(ids) == qa
    assert not set(tok.encode(qa)) & set(tok.special_tokens.values())

    tok.save(tmp_path / "wp.json")
    loaded = WordPieceTokenizer.load(tmp_path / "wp.json")
    assert loaded.tokens == tok.tokens
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode(qa, allowed_special="all") == ids


def test_parallel_counting_gives_same_tokenizer():
    assert _trained(num_workers=2, piece_chars=16).tokens == _trained().tokens
