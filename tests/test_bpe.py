import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from webbot.tokenizer import BPETokenizer


@pytest.fixture(params=["python", "rust"])
def backend(request):
    return request.param

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "def add_three(a, b, c):\n    return a + b + c\n",
]


def test_vocab_size_matches_request(backend):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    # vocab_size counts everything: bytes + merges + special tokens, with
    # ids 0..279 and no gaps.
    assert len(tok.vocab) == 280
    assert sorted(tok.vocab.values()) == list(range(280))
    assert len(tok.merges) == 280 - 256 - len(BPETokenizer.SPECIAL_TOKENS)


def test_eot_inserted_between_documents(backend):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    eot_id = tok.special_tokens[tok.EOT_TOKEN]

    ids = tok.encode_with_eot(CORPUS)

    assert ids.count(eot_id) == len(CORPUS)
    assert ids[-1] == eot_id
    assert ids == tok.encode(CORPUS[0]) + [eot_id] + tok.encode(CORPUS[1]) + [
        eot_id
    ] + tok.encode(CORPUS[2]) + [eot_id]


def test_eot_round_trips_through_decode(backend):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    eot_id = tok.special_tokens[tok.EOT_TOKEN]
    assert tok.decode([eot_id]) == tok.EOT_TOKEN


def test_encode_decode_roundtrip(backend):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    for text in CORPUS + ["def unseen(x):\n    return x * 2\n", "日本語もOK?"]:
        assert tok.decode(tok.encode(text)) == text


def test_common_substring_gets_own_token(backend):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    ids = tok.encode("    return a + b\n")
    assert len(ids) < len("    return a + b\n")


def test_save_and_load_roundtrip(backend, tmp_path):
    tok = BPETokenizer(backend)
    tok.train(CORPUS, vocab_size=280)
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = BPETokenizer.load(path, backend)
    text = "def add(a, b):\n    return a + b\n"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(tok.encode(text)) == text


def test_vocab_size_below_256_rejected(backend):
    tok = BPETokenizer(backend)
    try:
        tok.train(CORPUS, vocab_size=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for vocab_size < 256")


JA_CORPUS = [
    "むかしむかし、ある国に王様がいました。王様は毎晩、物語を聞きました。",
    "王様は言いました。「今夜も話をしておくれ。」",
    "ドラゴンは王様の城にやってきました。すごーい、と子どもたちは言いました。",
]


def test_japanese_split_at_script_boundaries():
    tok = BPETokenizer()
    assert tok._pretokenize("むかし、王様がドラゴンを見た。") == [
        "むかし", "、", "王様", "が", "ドラゴン", "を", "見", "た", "。",
    ]


def test_long_vowel_mark_stays_with_kana():
    tok = BPETokenizer()
    assert tok._pretokenize("すごーい") == ["すごーい"]
    assert tok._pretokenize("スーパー") == ["スーパー"]


def test_japanese_repeated_word_gets_single_token(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=300)
    # "王様" appears in every document, so BPE should merge its 6 UTF-8 bytes
    # into one token.
    assert len(tok.encode("王様")) == 1


def test_learned_tokens_never_cross_script_boundaries(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=400)
    for token_id in range(256, len(tok.merges) + 256):
        text = tok.decode([token_id])
        if "�" in text:  # partial UTF-8 sequence; can't pretokenize
            continue
        assert tok._pretokenize(text) == [text], text


def test_japanese_roundtrip(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=300)
    for text in JA_CORPUS + ["　全角スペースとｶﾀｶﾅと１２３。\n\n次の夜。"]:
        assert tok.decode(tok.encode(text)) == text


def _reference_merges(tok, texts, vocab_size):
    """Naive BPE: recount every pair from scratch before each merge."""
    word_freqs = {}
    for text in texts:
        for token in tok._pretokenize(text):
            word = tok._to_byte_symbols(token)
            word_freqs[word] = word_freqs.get(word, 0) + 1
    merges = []
    for _ in range(vocab_size - 256 - len(BPETokenizer.SPECIAL_TOKENS)):
        counts = tok._get_pair_counts(word_freqs)
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        word_freqs = {
            tok._merge_word(w, best, "".join(best)): f
            for w, f in word_freqs.items()
        }
        merges.append(best)
    return merges


def test_incremental_training_matches_full_recount(backend):
    import random

    rng = random.Random(0)
    # Small alphabet -> many overlapping repeats ("aaaa") and count ties,
    # which are the cases where incremental updates are easiest to get wrong.
    random_texts = [
        " ".join(
            "".join(rng.choice("abあい") for _ in range(rng.randint(1, 8)))
            for _ in range(20)
        )
        for _ in range(10)
    ]
    for texts in (CORPUS, JA_CORPUS, random_texts, ["aaaaaaa aaa aaaa"]):
        tok = BPETokenizer(backend)
        tok.train(texts, vocab_size=400)
        assert tok.merges == _reference_merges(BPETokenizer("python"), texts, 400)


# Text full of chunk-boundary hazards: contractions ("'ll" split as "'l"),
# whitespace runs whose split depends on the next char, leading-space tokens,
# and mixed scripts.
TRICKY_TEXT = (
    "We'll see, they're here!!  \n\n\n  むかし  王様は言いました。「すごーい!」\n"
    "   スーパードラゴン123と１２３ ｶﾀｶﾅ　全角 I'd've... \t\n end   "
)


def _split(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_chunked_pretokenize_matches_whole_text_for_every_chunk_size():
    tok = BPETokenizer()
    expected = tok._pretokenize(TRICKY_TEXT)
    for size in range(1, len(TRICKY_TEXT) + 1):
        got = list(tok._pretokenize_chunks(_split(TRICKY_TEXT, size)))
        assert got == expected, size


def test_training_on_chunked_generator_matches_whole_strings(backend):
    texts = JA_CORPUS + [TRICKY_TEXT]
    whole = BPETokenizer(backend)
    whole.train(texts, vocab_size=400)

    chunked = BPETokenizer(backend)
    # A generator of documents, each itself a generator of 7-char chunks:
    # the corpus is never held as full strings.
    chunked.train((iter(_split(t, 7)) for t in texts), vocab_size=400)

    assert chunked.merges == whole.merges
    assert chunked.vocab == whole.vocab


def test_encode_chunks_matches_encode(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS + [TRICKY_TEXT], vocab_size=400)
    for size in (1, 3, 7, 50):
        assert list(tok.encode_chunks(_split(TRICKY_TEXT, size))) == tok.encode(
            TRICKY_TEXT
        )


def test_safe_pieces_pretokenize_like_whole_text():
    from webbot.tokenizer.bpe import _safe_pieces

    tok = BPETokenizer()
    expected = tok._pretokenize(TRICKY_TEXT)
    for size in range(1, len(TRICKY_TEXT) + 1):
        pieces = list(_safe_pieces(_split(TRICKY_TEXT, size), target_chars=size))
        assert "".join(pieces) == TRICKY_TEXT
        got = [t for piece in pieces for t in tok._pretokenize(piece)]
        assert got == expected, size
    # The test only means something if the text actually gets split.
    assert len(list(_safe_pieces([TRICKY_TEXT], target_chars=1))) > 10


def test_parallel_training_matches_single_process(backend):
    texts = JA_CORPUS * 5 + [TRICKY_TEXT]
    single = BPETokenizer(backend)
    single.train(texts, vocab_size=400)

    parallel = BPETokenizer(backend)
    parallel.train(
        (iter(_split(t, 7)) for t in texts),
        vocab_size=400,
        num_workers=2,
        piece_chars=16,
    )

    assert parallel.merges == single.merges
    assert parallel.vocab == single.vocab


def test_bundle_groups_small_pieces_without_losing_any():
    from webbot.tokenizer.bpe import _bundle

    pieces = [f"話{i}" * (i % 5 + 1) for i in range(100)]
    bundles = list(_bundle(pieces, target_chars=60))  # 600 chars -> ~10 bundles
    assert [p for b in bundles for p in b] == pieces
    assert len(bundles) < len(pieces) / 5  # far fewer tasks than pieces
    assert all(sum(map(len, b)) >= 60 for b in bundles[:-1])


def test_parallel_training_on_many_short_docs_matches_single_process(backend):
    # Many tiny documents are bundled into few worker tasks; documents must
    # still be pretokenized separately (never glued together).
    texts = [f"{w}は城に行きました。" for w in ("王様", "猫", "ドラゴン", "旅人")] * 50
    single = BPETokenizer(backend)
    single.train(texts, vocab_size=320)
    parallel = BPETokenizer(backend)
    parallel.train(texts, vocab_size=320, num_workers=2, piece_chars=200)
    assert parallel.merges == single.merges


def test_count_ties_break_on_symbol_strings_not_byte_values(backend):
    # " b" and "ab" each occur once, so (Ġ, b) and (a, b) tie. The tie goes
    # to the lexicographically larger pair of *symbol strings*: "Ġ" (the
    # printable stand-in for the space byte, U+0120) > "a", although the raw
    # byte 0x20 < 0x61.
    tok = BPETokenizer(backend)
    tok.train(["ab", " b"], vocab_size=256 + len(tok.SPECIAL_TOKENS) + 1)
    assert tok.merges == [("Ġ", "b")]


def _random_corpus(seed, n_docs, alphabet):
    import random

    rng = random.Random(seed)
    return [
        " ".join(
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            for _ in range(rng.randint(1, 60))
        )
        for _ in range(n_docs)
    ]


def test_rust_training_matches_python_reference():
    # Large enough for thousands of merges, with heavy count ties, repeated
    # symbols ("aaaa") and multi-byte characters; the last corpus runs out of
    # pairs before reaching the requested vocab size.
    corpora = [
        (_random_corpus(1, 300, "abcあいう漢字カナ。ー"), 2000),
        (_random_corpus(2, 200, "aab"), 1000),
        (JA_CORPUS + [TRICKY_TEXT] + CORPUS, 3000),
    ]
    for texts, vocab_size in corpora:
        py = BPETokenizer("python")
        py.train(texts, vocab_size=vocab_size)
        rs = BPETokenizer("rust")
        rs.train(texts, vocab_size=vocab_size)
        assert rs.merges == py.merges
        assert rs.vocab == py.vocab
    assert len(rs.merges) < 3000 - 256  # merging stopped early


def test_rust_encode_matches_python_reference():
    texts = _random_corpus(3, 300, "abcあいう漢字カナ。ー")
    tok = BPETokenizer("python")
    tok.train(texts[:200], vocab_size=1500)
    unseen = texts[200:] + [TRICKY_TEXT, "未知の文字列🐍とemoji、改行\n\n"]
    expected = [tok.encode(t) for t in unseen]

    rs = BPETokenizer("rust")
    rs.merges, rs.ranks, rs.vocab = tok.merges, tok.ranks, tok.vocab
    assert [rs.encode(t) for t in unseen] == expected
    # Encoding again is served from the word cache.
    assert [rs.encode(t) for t in unseen] == expected


def test_rust_encoder_cache_eviction_keeps_results():
    import webbot_bpe

    tok = BPETokenizer("python")
    tok.train(JA_CORPUS * 3, vocab_size=400)
    merges = [(tok.vocab[a], tok.vocab[b], tok.vocab[a + b]) for a, b in tok.merges]
    tiny = webbot_bpe.Encoder(merges, cache_limit=2)  # cleared every 2 words
    words = tok._pretokenize(" ".join(JA_CORPUS) * 2)
    expected = [i for w in words for i in tok.encode(w)]
    assert tiny.encode_words(words) == expected
    assert tiny.encode_words(words) == expected


QA_TEXT = (
    "<|question|>王様の城はどこ？<|search|>王様 城<|/search|>"
    "<|doc|>王様の城は丘の上にあります。<|/doc|><|doc|>城には塔が3つ。<|/doc|>"
    "<|answer|>丘の上<|endoftext|>"
)


def _reference_encode_special(tok, text, allowed):
    """Naive reference: scan for the earliest allowed special token with
    str.find, encode the plain text before it, emit the special id, repeat."""
    ids = []
    while True:
        hits = [(text.find(t), -len(t), t) for t in allowed if t in text]
        if not hits:
            return ids + (tok.encode(text) if text else [])
        pos, _, token = min(hits)  # earliest; longest on a tie
        if pos:
            ids += tok.encode(text[:pos])
        ids.append(tok.special_tokens[token])
        text = text[pos + len(token):]


def test_special_tokens_get_ids_right_after_merges(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=400)
    first = 256 + len(tok.merges)
    assert tok.special_tokens == {
        t: first + i for i, t in enumerate(BPETokenizer.SPECIAL_TOKENS)
    }
    assert tok.special_tokens[tok.EOT_TOKEN] == first  # same slot as before
    for t, i in tok.special_tokens.items():
        assert tok.decode([i]) == t


def test_special_token_strings_are_plain_text_by_default(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS + [QA_TEXT], vocab_size=400)
    ids = tok.encode(QA_TEXT)
    assert not set(ids) & set(tok.special_tokens.values())
    assert tok.decode(ids) == QA_TEXT
    assert list(tok.encode_chunks(_split(QA_TEXT, 5))) == ids


def test_allowed_special_tokens_become_single_ids(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=400)
    ids = tok.encode(QA_TEXT, allowed_special="all")
    assert ids == _reference_encode_special(tok, QA_TEXT, tok.special_tokens)
    sp = tok.special_tokens
    assert ids[0] == sp["<|question|>"] and ids[-1] == sp[tok.EOT_TOKEN]
    assert ids.count(sp["<|doc|>"]) == 2 and ids.count(sp["<|/doc|>"]) == 2
    assert tok.decode(ids) == QA_TEXT


def test_only_listed_special_tokens_are_allowed(backend):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=400)
    allowed = {"<|doc|>", "<|/doc|>"}
    ids = tok.encode(QA_TEXT, allowed_special=allowed)
    assert ids == _reference_encode_special(tok, QA_TEXT, allowed)
    emitted = set(ids) & set(tok.special_tokens.values())
    assert emitted == {tok.special_tokens[t] for t in allowed}
    assert tok.decode(ids) == QA_TEXT
    with pytest.raises(ValueError):
        tok.encode(QA_TEXT, allowed_special={"<|nope|>"})


def test_special_tokens_survive_save_and_load(backend, tmp_path):
    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=400)
    tok.save(tmp_path / "tok.json")
    loaded = BPETokenizer.load(tmp_path / "tok.json", backend)
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.encode(QA_TEXT, allowed_special="all") == tok.encode(
        QA_TEXT, allowed_special="all"
    )


def test_old_tokenizer_file_without_special_list_loads_with_eot_only(backend, tmp_path):
    import json

    tok = BPETokenizer(backend)
    tok.train(JA_CORPUS, vocab_size=300)
    # The format saved before special tokens existed: EOT right after merges.
    vocab = {s: i for s, i in tok.vocab.items() if i < 256 + len(tok.merges)}
    vocab[tok.EOT_TOKEN] = len(vocab)
    (tmp_path / "old.json").write_text(
        json.dumps({"merges": tok.merges, "vocab": vocab}), encoding="utf-8"
    )
    old = BPETokenizer.load(tmp_path / "old.json", backend)
    assert old.special_tokens == {tok.EOT_TOKEN: 256 + len(tok.merges)}
    assert old.encode_with_eot(JA_CORPUS) == tok.encode_with_eot(JA_CORPUS)
