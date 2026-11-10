from __future__ import annotations

import heapq
import json
import re
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

try:  # native training / encoding (built from rust/ with maturin)
    import webbot_bpe as _native
except ImportError:
    _native = None

# Character classes for pre-tokenization. Japanese has no spaces between
# words, so script boundaries (hiragana / katakana / kanji / punctuation)
# stand in for word boundaries. "ー" belongs to both kana classes so that
# long vowels stay attached ("すごーい", "ドラゴン").
_LATIN = "A-Za-zＡ-Ｚａ-ｚ"
_DIGIT = "0-9０-９"
_HIRAGANA = "ぁ-ゖゝゞー"
_KATAKANA = "ァ-ヺーヽヾｦ-ﾟ"
_KANJI = "一-鿿㐀-䶿々〆"

# GPT-2's pre-tokenization pattern, extended with Japanese script classes.
_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    rf"""| ?[{_LATIN}]+| ?[{_DIGIT}]+"""
    rf"""| ?[{_HIRAGANA}]+| ?[{_KATAKANA}]+| ?[{_KANJI}]+"""
    rf"""| ?[^\s{_LATIN}{_DIGIT}{_HIRAGANA}{_KATAKANA}{_KANJI}]+"""
    r"""|\s+(?!\S)|\s+"""
)

_CLASS_PATTERNS = [
    re.compile(f"[{cls}]") for cls in (_LATIN, _DIGIT, _HIRAGANA, _KATAKANA, _KANJI)
]
_SPACE = re.compile(r"\s")


def _char_classes(c: str) -> frozenset[int]:
    """Indices of the pretokenizer character classes c belongs to (5 = the
    catch-all symbol class). "ー" belongs to two."""
    found = frozenset(i for i, p in enumerate(_CLASS_PATTERNS) if p.match(c))
    return found or frozenset((len(_CLASS_PATTERNS),))


def _is_safe_split(text: str, i: int) -> bool:
    """True if pretokenizing text[:i] and text[i:] separately gives exactly
    the pretokens of text.

    The pattern has no lookbehind, so matching from i is unaffected by what
    precedes it; it only has to be guaranteed that a pretoken boundary falls
    at i. That holds when both neighbours are non-space and share no
    character class (so no run can span them), except after "'", where a
    contraction such as "'ll" may be in progress.
    """
    a, b = text[i - 1], text[i]
    if a == "'" or _SPACE.match(a) or _SPACE.match(b):
        return False
    return not (_char_classes(a) & _char_classes(b))


def _safe_pieces(chunks: Iterable[str], target_chars: int) -> Iterator[str]:
    """Regroup one document's chunks into pieces of about target_chars that
    can be pretokenized independently (split only at _is_safe_split points).
    A piece grows past target_chars when no safe point is found."""
    buf = ""
    scan_from = target_chars  # cut at the first safe point at or after this
    for chunk in chunks:
        buf += chunk
        while len(buf) > scan_from:
            cut = next(
                (
                    i
                    for i in range(max(scan_from, 1), len(buf))
                    if _is_safe_split(buf, i)
                ),
                None,
            )
            if cut is None:
                # Wait for more text; don't rescan what was already checked.
                scan_from = len(buf)
                break
            yield buf[:cut]
            buf = buf[cut:]
            scan_from = target_chars
    if buf:
        yield buf


def _count_pretokens(texts: list[str]) -> Counter[str]:
    """Worker task for parallel training (module-level so it can be pickled).
    Each text is pretokenized on its own: they are independent pieces."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_SPLIT_PATTERN.findall(text))
    return counts


def _bundle(pieces: Iterable[str], target_chars: int) -> Iterator[list[str]]:
    """Group pieces into lists of about target_chars, so a corpus of many
    short documents is sent to workers in a few large tasks rather than
    one task (and one round of pickling) per document."""
    bundle: list[str] = []
    size = 0
    for piece in pieces:
        bundle.append(piece)
        size += len(piece)
        if size >= target_chars:
            yield bundle
            bundle, size = [], 0
    if bundle:
        yield bundle


def _bytes_to_unicode() -> dict[int, str]:
    """Map every byte value to a printable unicode char (GPT-2's byte encoder).

    Keeps merge rules and vocab entries as plain strings while still being
    able to round-trip arbitrary bytes, including ones with no printable
    glyph (control chars, etc.).
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class _Desc:
    """Heap key that orders pairs in reverse, so heapq (a min-heap) pops the
    lexicographically largest pair first among equal counts."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[str, str]) -> None:
        self.pair = pair

    def __lt__(self, other: "_Desc") -> bool:
        return self.pair > other.pair


class BPETokenizer:
    """Byte-level BPE tokenizer, trained from scratch (GPT-2 style).

    backend selects how merges are learned and applied: "rust" uses the
    native webbot_bpe module, "python" the pure-Python reference
    implementation below, and "auto" the native module when it is installed.
    Both give identical merges and ids.
    """

    # Document-boundary marker for pretraining data, also ends an answer.
    EOT_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    # Special tokens get ids right after the learned merges, in this order.
    # They are never produced by BPE merges; encode() emits them only when
    # explicitly allowed, so the same strings inside ordinary text (e.g. a
    # Wikipedia article) are encoded as plain text. Retrieval-QA layout:
    #   <|question|>Q<|search|>query<|/search|><|doc|>P1<|/doc|>...<|answer|>A<|endoftext|>
    SPECIAL_TOKENS = (
        EOT_TOKEN,
        PAD_TOKEN,  # batch padding; never a training target
        "<|question|>",  # the user's question follows
        "<|search|>",  # the model writes a search query ...
        "<|/search|>",  # ... and ends it here (retrieval runs)
        "<|doc|>",  # one retrieved paragraph follows ...
        "<|/doc|>",  # ... and ends here
        "<|answer|>",  # the answer follows, ended by EOT
    )

    # Pretokens sent to the native encoder per call in encode_chunks.
    _NATIVE_BATCH = 4096

    def __init__(self, backend: str = "auto") -> None:
        if backend not in ("auto", "rust", "python"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "rust" and _native is None:
            raise ImportError("webbot_bpe is not installed (build rust/ with maturin)")
        self.backend = backend
        self._encoder = None  # native encoder, built lazily from the merges
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.merges: list[tuple[str, str]] = []
        self.ranks: dict[tuple[str, str], int] = {}
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        self.special_tokens: dict[str, int] = {}
        self._special_ids: dict[int, str] = {}

    @property
    def _use_native(self) -> bool:
        return self.backend != "python" and _native is not None

    def _native_encoder(self):
        if self._encoder is None:
            self._encoder = _native.Encoder(
                [
                    (self.vocab[a], self.vocab[b], self.vocab[a + b])
                    for a, b in self.merges
                ]
            )
        return self._encoder

    def _pretokenize(self, text: str) -> list[str]:
        return _SPLIT_PATTERN.findall(text)

    @staticmethod
    def _pretokenize_chunks(chunks: Iterable[str]) -> Iterator[str]:
        """Pretokenize one document given as consecutive text chunks.

        Yields exactly what _pretokenize("".join(chunks)) would, without ever
        holding the whole document. The last two matches of each buffer are
        held back and re-matched with the next chunk: only they can change
        once more text arrives (a match that ends at the buffer edge may keep
        growing, and a contraction like "'ll" cut to "'l" shows up as two
        short matches). Every earlier match starts at least 3 chars before
        the edge, so its alternatives and lookahead see the same text either
        way.
        """
        buf = ""
        for chunk in chunks:
            buf += chunk
            held: deque[re.Match[str]] = deque()
            for m in _SPLIT_PATTERN.finditer(buf):
                held.append(m)
                if len(held) > 2:
                    yield held.popleft().group()
            if held:
                buf = buf[held[0].start():]
        yield from _SPLIT_PATTERN.findall(buf)

    def _to_byte_symbols(self, token: str) -> tuple[str, ...]:
        return tuple(self.byte_encoder[b] for b in token.encode("utf-8"))

    @staticmethod
    def _get_pair_counts(
        word_freqs: dict[tuple[str, ...], int]
    ) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for word, freq in word_freqs.items():
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        return counts

    @staticmethod
    def _merge_word(
        word: tuple[str, ...], pair: tuple[str, str], merged: str
    ) -> tuple[str, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    @staticmethod
    def _count_parallel(
        docs: Iterable[Iterable[str]], num_workers: int, piece_chars: int
    ) -> Counter[str]:
        pieces = (
            piece for chunks in docs for piece in _safe_pieces(chunks, piece_chars)
        )
        token_freqs: Counter[str] = Counter()
        # Keep at most 2 bundles per worker in flight so a large corpus is
        # never queued into memory ahead of the workers.
        max_pending = 2 * num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            pending: set[Future[Counter[str]]] = set()
            for bundle in _bundle(pieces, piece_chars):
                pending.add(pool.submit(_count_pretokens, bundle))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        token_freqs.update(f.result())
            for f in pending:
                token_freqs.update(f.result())
        return token_freqs

    def train(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int,
        verbose: bool = False,
        num_workers: int = 1,
        piece_chars: int = 1 << 20,
    ) -> None:
        """Learn merges from documents.

        Each document is either a str or an iterable of str chunks (e.g. a
        large file read piece by piece), and texts itself may be a generator,
        so the corpus is streamed: only the counts of distinct pretokens are
        kept in memory, never the corpus text.

        With num_workers > 1, pretokenizing and counting run in worker
        processes on pieces of about piece_chars characters. The learned
        merges are identical to the single-process result.

        vocab_size is the size of the final vocab, special tokens included:
        256 byte symbols + merges + SPECIAL_TOKENS.
        """
        base_symbols = [self.byte_encoder[b] for b in range(256)]
        min_size = len(base_symbols) + len(self.SPECIAL_TOKENS)
        if vocab_size < min_size:
            raise ValueError(f"vocab_size must be >= {min_size}")

        docs = ((doc,) if isinstance(doc, str) else doc for doc in texts)
        if num_workers > 1:
            token_freqs = self._count_parallel(docs, num_workers, piece_chars)
        else:
            token_freqs = Counter()
            for chunks in docs:
                token_freqs.update(self._pretokenize_chunks(chunks))

        num_merges = vocab_size - min_size
        if self._use_native:
            learned = _native.train_bpe(
                list(token_freqs.items()), base_symbols, num_merges
            )
            del token_freqs
            self.merges = [(a, b) for a, b, _ in learned]
            if verbose:
                for i, (a, b, count) in enumerate(learned):
                    self._print_merge(i, num_merges, (a, b), count)
        else:
            self.merges = self._learn_merges(token_freqs, num_merges, verbose)

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._build_vocab(base_symbols)
        self._encoder = None

    @staticmethod
    def _print_merge(i: int, num_merges: int, pair: tuple[str, str], count: int) -> None:
        print(f"merge {i + 1}/{num_merges}: {pair} -> {''.join(pair)!r} (count={count})")

    def _learn_merges(
        self, token_freqs: Counter[str], num_merges: int, verbose: bool
    ) -> list[tuple[str, str]]:
        """Pure-Python reference implementation of the merge loop."""
        word_freqs: dict[tuple[str, ...], int] = {}
        for token, freq in token_freqs.items():
            word = self._to_byte_symbols(token)
            word_freqs[word] = word_freqs.get(word, 0) + freq
        del token_freqs

        words = list(word_freqs)
        freqs = [word_freqs[w] for w in words]

        # Pair counts are computed once, then kept up to date incrementally:
        # a merge only touches the words that contain the merged pair, so only
        # those words' pairs are subtracted and re-added. pair_to_words is an
        # index from pair to candidate words (it may hold stale entries for
        # words that no longer contain the pair; merging those is a no-op).
        pair_counts = self._get_pair_counts(word_freqs)
        pair_to_words: dict[tuple[str, str], set[int]] = {}
        for idx, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_to_words.setdefault(pair, set()).add(idx)

        # Max-heap over (count, pair) with lazy deletion: an entry is valid
        # only if its count still matches pair_counts. Ties break toward the
        # lexicographically larger pair, same as max(key=(count, pair)).
        heap = [(-count, _Desc(pair)) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        for i in range(num_merges):
            best_pair = None
            while heap:
                neg_count, desc = heapq.heappop(heap)
                if pair_counts.get(desc.pair) == -neg_count:
                    best_pair, best_count = desc.pair, -neg_count
                    break
            if best_pair is None:
                break

            merged_symbol = "".join(best_pair)
            changed: set[tuple[str, str]] = set()
            for idx in pair_to_words.pop(best_pair, ()):
                word = words[idx]
                new_word = self._merge_word(word, best_pair, merged_symbol)
                if new_word == word:
                    continue
                freq = freqs[idx]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= freq
                    changed.add(pair)
                for pair in zip(new_word, new_word[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    pair_to_words.setdefault(pair, set()).add(idx)
                    changed.add(pair)
                words[idx] = new_word

            for pair in changed:
                count = pair_counts[pair]
                if count > 0:
                    heapq.heappush(heap, (-count, _Desc(pair)))
                else:
                    del pair_counts[pair]

            merges.append(best_pair)
            if verbose:
                self._print_merge(i, num_merges, best_pair, best_count)
        return merges

    def _build_vocab(self, base_symbols: list[str]) -> None:
        self.vocab = {sym: i for i, sym in enumerate(base_symbols)}
        for a, b in self.merges:
            self.vocab["".join((a, b))] = len(self.vocab)
        for token in self.SPECIAL_TOKENS:
            # Pretokenization splits "<|", the name and "|>" apart, so no
            # merge can ever produce a whole special token.
            assert token not in self.vocab, token
            self.vocab[token] = len(self.vocab)
        self._set_special_tokens(self.SPECIAL_TOKENS)
        self.inverse_vocab = {i: sym for sym, i in self.vocab.items()}

    def _set_special_tokens(self, tokens: Iterable[str]) -> None:
        self.special_tokens = {t: self.vocab[t] for t in tokens}
        self._special_ids = {i: t for t, i in self.special_tokens.items()}

    def _allowed_set(self, allowed_special: Iterable[str] | str) -> set[str]:
        if allowed_special == "all":
            return set(self.special_tokens)
        allowed = set(allowed_special)
        unknown = allowed - self.special_tokens.keys()
        if unknown:
            raise ValueError(f"not special tokens: {sorted(unknown)}")
        return allowed

    def _bpe_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            candidate = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if candidate not in self.ranks:
                break
            word = self._merge_word(word, candidate, "".join(candidate))
        return word

    def encode(
        self, text: str, allowed_special: Iterable[str] | str = ()
    ) -> list[int]:
        """Encode text to ids.

        Occurrences of the special tokens listed in allowed_special (or of
        every special token, with "all") become their special ids; the text
        between them is encoded on its own. Special-token strings that are
        not allowed are encoded as ordinary text.
        """
        allowed = self._allowed_set(allowed_special)
        if not allowed:
            return self._encode_ordinary(text)
        # Longest first, so a token that is a prefix of another can't win.
        pattern = "(" + "|".join(
            re.escape(t) for t in sorted(allowed, key=len, reverse=True)
        ) + ")"
        ids: list[int] = []
        for i, part in enumerate(re.split(pattern, text)):
            if i % 2:  # captured separator = a special token
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        if self._use_native:
            return self._native_encoder().encode_words(self._pretokenize(text))
        ids = []
        for token in self._pretokenize(text):
            word = self._to_byte_symbols(token)
            for symbol in self._bpe_word(word):
                ids.append(self.vocab[symbol])
        return ids

    def encode_chunks(self, chunks: Iterable[str]) -> Iterator[int]:
        """Lazily encode one document given as consecutive text chunks.

        Same ids as encode("".join(chunks)), without materializing the text
        or the full id list; callers can write ids out as they arrive.
        """
        if self._use_native:
            encoder = self._native_encoder()
            batch: list[str] = []
            for token in self._pretokenize_chunks(chunks):
                batch.append(token)
                if len(batch) >= self._NATIVE_BATCH:
                    yield from encoder.encode_words(batch)
                    batch = []
            yield from encoder.encode_words(batch)
            return
        for token in self._pretokenize_chunks(chunks):
            for symbol in self._bpe_word(self._to_byte_symbols(token)):
                yield self.vocab[symbol]

    def encode_with_eot(self, texts: list[str]) -> list[int]:
        """Encode multiple documents into one id stream, EOT-separated.

        This is the shape pretraining needs: a flat sequence of token ids
        with an explicit boundary marker between otherwise-unrelated
        documents, so the model learns "this is the end" rather than
        blending the tail of one snippet into the head of the next.
        """
        eot_id = self.special_tokens[self.EOT_TOKEN]
        ids: list[int] = []
        for text in texts:
            ids.extend(self.encode(text))
            ids.append(eot_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_seq = bytearray()
        for i in ids:
            special = self._special_ids.get(i)
            if special is not None:
                byte_seq.extend(special.encode("utf-8"))
            else:
                byte_seq.extend(self.byte_decoder[ch] for ch in self.inverse_vocab[i])
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        data = {
            "merges": [list(pair) for pair in self.merges],
            "vocab": self.vocab,
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, backend: str = "auto") -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(backend)
        tok.merges = [tuple(pair) for pair in data["merges"]]
        tok.ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.vocab = data["vocab"]
        tok.inverse_vocab = {i: sym for sym, i in tok.vocab.items()}
        # Files written before special tokens were saved have only EOT.
        tok._set_special_tokens(data.get("special_tokens", [tok.EOT_TOKEN]))
        return tok
