from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .base import Tokenizer
from .pretokenize import count_pretokens

try:  # native training / encoding (built from rust/ with maturin)
    import webbot_bpe as _native
except ImportError:
    _native = None


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


class BPETokenizer(Tokenizer):
    """Byte-level BPE tokenizer, trained from scratch (GPT-2 style).

    backend selects how merges are learned and applied: "rust" uses the
    native webbot_bpe module, "python" the pure-Python reference
    implementation below, and "auto" the native module when it is installed.
    Both give identical merges and ids.
    """

    def __init__(self, backend: str = "auto") -> None:
        super().__init__()
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

    def train(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int,
        verbose: bool = False,
        num_workers: int = 1,
        piece_chars: int = 1 << 20,
    ) -> None:
        """Learn merges from documents (see count_pretokens for how texts,
        num_workers and piece_chars are handled; the learned merges do not
        depend on num_workers).

        vocab_size is the size of the final vocab, special tokens included:
        256 byte symbols + merges + SPECIAL_TOKENS.
        """
        base_symbols = [self.byte_encoder[b] for b in range(256)]
        min_size = len(base_symbols) + len(self.SPECIAL_TOKENS)
        if vocab_size < min_size:
            raise ValueError(f"vocab_size must be >= {min_size}")

        token_freqs = count_pretokens(texts, num_workers, piece_chars)

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
        self._set_special_tokens({t: self.vocab[t] for t in self.SPECIAL_TOKENS})
        self.inverse_vocab = {i: sym for sym, i in self.vocab.items()}

    def _bpe_word(self, word: tuple[str, ...]) -> tuple[str, ...]:
        while len(word) > 1:
            pairs = list(zip(word, word[1:]))
            candidate = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if candidate not in self.ranks:
                break
            word = self._merge_word(word, candidate, "".join(candidate))
        return word

    def __len__(self) -> int:
        return len(self.vocab)

    def _encode_pretokens(self, pretokens: list[str]) -> list[int]:
        if self._use_native:
            return self._native_encoder().encode_words(pretokens)
        ids = []
        for token in pretokens:
            for symbol in self._bpe_word(self._to_byte_symbols(token)):
                ids.append(self.vocab[symbol])
        return ids

    def _token_bytes(self, token_id: int) -> bytes:
        return bytes(self.byte_decoder[ch] for ch in self.inverse_vocab[token_id])

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
        tok._set_special_tokens(
            {t: tok.vocab[t] for t in data.get("special_tokens", [tok.EOT_TOKEN])}
        )
        return tok
