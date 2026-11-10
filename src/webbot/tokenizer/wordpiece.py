"""WordPiece tokenizer (BERT style), on top of the shared pre-tokenization.

Training merges adjacent symbols like BPE, but picks the pair with the
highest score count(a, b) / (count(a) * count(b)) rather than the most
frequent pair, so pairs whose parts rarely occur apart are merged first.
Pieces that continue a word (not at its start) are separate tokens, shown
with a "##" prefix. Encoding does not replay merges: each word is split
greedily, longest vocabulary match first, from the left.

Characters outside the alphabet fall back to their UTF-8 bytes (ids
0..255), so any text can be encoded and decoded losslessly.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .base import Tokenizer
from .pretokenize import count_pretokens, required_chars

# A symbol is (continues_word, text). Keeping the flag separate (instead of
# a "##" prefix in the string) stays unambiguous for text that contains "#".
Symbol = tuple[bool, str]
Pair = tuple[Symbol, Symbol]


def display(sym: Symbol) -> str:
    cont, text = sym
    return "##" + text if cont else text


def split_word(word: str, alphabet: set[str]) -> list[Symbol | None]:
    """Initial symbols of a word: one per character, continuation-marked
    after the first; None for a character outside the alphabet (a barrier
    that never merges)."""
    return [(i > 0, ch) if ch in alphabet else None for i, ch in enumerate(word)]


def merge_symbols(a: Symbol, b: Symbol) -> Symbol:
    return (a[0], a[1] + b[1])


def merge_word(
    word: tuple[Symbol | None, ...], pair: Pair, merged: Symbol
) -> tuple[Symbol | None, ...]:
    """Replace each non-overlapping occurrence of pair, left to right."""
    out = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def _pairs(word: tuple[Symbol | None, ...]) -> Iterable[Pair]:
    for a, b in zip(word, word[1:]):
        if a is not None and b is not None:
            yield (a, b)


class _Desc:
    """Heap key that orders pairs in reverse, so heapq (a min-heap) pops the
    largest pair first among equal scores."""

    __slots__ = ("pair",)

    def __init__(self, pair: Pair) -> None:
        self.pair = pair

    def __lt__(self, other: "_Desc") -> bool:
        return self.pair > other.pair


def learn_merges(
    word_freqs: dict[tuple[Symbol | None, ...], int], num_merges: int
) -> list[Pair]:
    """Up to num_merges merges, each the pair with the highest
    count(a, b) / (count(a) * count(b)); ties go to the larger pair.

    Counts are kept up to date incrementally (only words containing the
    merged pair change). A pair's score also depends on its symbols'
    counts, so after a merge every pair containing one of the two merged
    symbols (whose counts dropped) is re-scored. The heap holds
    (score, pair) with lazy deletion: an entry is valid only if it matches
    the pair's current score.
    """
    words = list(word_freqs)
    freqs = [word_freqs[w] for w in words]
    sym_count: Counter[Symbol] = Counter()
    pair_count: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[int]] = {}
    sym_to_pairs: dict[Symbol, set[Pair]] = {}
    for idx, (word, f) in enumerate(zip(words, freqs)):
        for s in word:
            if s is not None:
                sym_count[s] += f
        for p in _pairs(word):
            pair_count[p] += f
            pair_to_words.setdefault(p, set()).add(idx)
            sym_to_pairs.setdefault(p[0], set()).add(p)
            sym_to_pairs.setdefault(p[1], set()).add(p)

    def score(p: Pair) -> float:
        return pair_count[p] / (sym_count[p[0]] * sym_count[p[1]])

    heap = [(-score(p), _Desc(p)) for p in pair_count]
    heapq.heapify(heap)

    merges: list[Pair] = []
    while len(merges) < num_merges:
        best = None
        while heap:
            neg, desc = heapq.heappop(heap)
            p = desc.pair
            if pair_count.get(p, 0) > 0 and score(p) == -neg:
                best = p
                break
        if best is None:
            break

        merged = merge_symbols(*best)
        changed: set[Pair] = set()
        for idx in pair_to_words.pop(best, ()):
            word = words[idx]
            new_word = merge_word(word, best, merged)
            if new_word == word:
                continue
            f = freqs[idx]
            n_merged = len(word) - len(new_word)
            sym_count[best[0]] -= n_merged * f
            sym_count[best[1]] -= n_merged * f
            sym_count[merged] += n_merged * f
            for p in _pairs(word):
                pair_count[p] -= f
                changed.add(p)
            for p in _pairs(new_word):
                pair_count[p] += f
                pair_to_words.setdefault(p, set()).add(idx)
                sym_to_pairs.setdefault(p[0], set()).add(p)
                sym_to_pairs.setdefault(p[1], set()).add(p)
                changed.add(p)
            words[idx] = new_word

        # The merged symbols' counts dropped: all their pairs score higher.
        # (The new symbol is new: merges apply everywhere, left to right, in
        # one order, so no other pair rebuilds it; _build would also reject
        # a duplicate piece.)
        for s in best:
            changed |= sym_to_pairs.get(s, set())
        for p in changed:
            if pair_count.get(p, 0) > 0:
                heapq.heappush(heap, (-score(p), _Desc(p)))
            else:
                pair_count.pop(p, None)
        merges.append(best)
    return merges


class WordPieceTokenizer(Tokenizer):
    """WordPiece with "##" continuation pieces and byte fallback.

    Ids: 0..255 are raw bytes (fallback), then the alphabet (single
    characters, in sorted order), then merged pieces in the order they were
    learned, then SPECIAL_TOKENS.
    """

    _CACHE_LIMIT = 1 << 20

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[Symbol] = []  # ids 256.. in order
        self.vocab: dict[Symbol, int] = {}
        self.max_len = 1
        self._cache: dict[str, list[int]] = {}

    def train(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int,
        num_workers: int = 1,
        piece_chars: int = 1 << 20,
        character_coverage: float = 0.9995,
    ) -> None:
        """Learn the vocabulary from documents (texts as for count_pretokens).

        vocab_size counts everything: 256 byte ids + alphabet + merged
        pieces + SPECIAL_TOKENS. The alphabet holds every character (in the
        word-initial and/or continuation form seen in the corpus) needed to
        reach character_coverage; it is an upper bound if merges run out.
        """
        budget = vocab_size - 256 - len(self.SPECIAL_TOKENS)
        if budget < 1:
            raise ValueError(f"vocab_size must be > {256 + len(self.SPECIAL_TOKENS)}")
        token_freqs = count_pretokens(texts, num_workers, piece_chars)
        alphabet = set(required_chars(token_freqs, character_coverage))
        word_freqs: dict[tuple[Symbol | None, ...], int] = {}
        for token, f in token_freqs.items():
            w = tuple(split_word(token, alphabet))
            word_freqs[w] = word_freqs.get(w, 0) + f
        del token_freqs

        initial = sorted({s for w in word_freqs for s in w if s is not None})
        if len(initial) > budget:
            raise ValueError(
                f"the alphabet needs {len(initial)} symbols for coverage "
                f"{character_coverage}, but only {budget} fit"
            )
        merges = learn_merges(word_freqs, budget - len(initial))
        self._build(initial + [merge_symbols(*p) for p in merges])

    def _build(self, tokens: list[Symbol], special_tokens: Iterable[str] | None = None) -> None:
        self.tokens = tokens
        self.vocab = {s: 256 + i for i, s in enumerate(tokens)}
        assert len(self.vocab) == len(tokens), "duplicate pieces"
        self.max_len = max((len(t) for _, t in tokens), default=1)
        specials = self.SPECIAL_TOKENS if special_tokens is None else special_tokens
        first = 256 + len(tokens)
        self._set_special_tokens({t: first + i for i, t in enumerate(specials)})
        self._cache = {}

    def __len__(self) -> int:
        return 256 + len(self.tokens) + len(self.special_tokens)

    def _encode_word(self, word: str) -> list[int]:
        """Greedy longest-match-first from the left; a character with no
        matching piece falls back to its bytes."""
        ids: list[int] = []
        i, n = 0, len(word)
        while i < n:
            cont = i > 0
            for j in range(min(n, i + self.max_len), i, -1):
                tid = self.vocab.get((cont, word[i:j]))
                if tid is not None:
                    ids.append(tid)
                    i = j
                    break
            else:
                ids.extend(word[i].encode("utf-8"))
                i += 1
        return ids

    def _encode_pretokens(self, pretokens: list[str]) -> list[int]:
        ids: list[int] = []
        for word in pretokens:
            cached = self._cache.get(word)
            if cached is None:
                cached = self._encode_word(word)
                if len(self._cache) >= self._CACHE_LIMIT:
                    self._cache.clear()
                self._cache[word] = cached
            ids.extend(cached)
        return ids

    def _token_bytes(self, token_id: int) -> bytes:
        if token_id < 256:
            return bytes((token_id,))
        return self.tokens[token_id - 256][1].encode("utf-8")

    def save(self, path: str | Path) -> None:
        data = {
            "type": "wordpiece",
            "tokens": [[text, cont] for cont, text in self.tokens],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordPieceTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok._build([(cont, text) for text, cont in data["tokens"]], data["special_tokens"])
        return tok
