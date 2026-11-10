"""Unigram language-model tokenizer (Kudo, 2018), as in SentencePiece's
"unigram" model type, on top of the shared pre-tokenization.

Each pretoken is segmented into pieces (substrings) so as to maximize the
sum of the pieces' log probabilities. Training starts from a large seed
vocabulary of frequent substrings and alternates EM (re-estimating piece
probabilities from expected counts) with pruning the pieces whose removal
costs the least likelihood, until the vocabulary is small enough.

Characters outside the vocabulary fall back to their UTF-8 bytes (ids
0..255), so any text can be encoded and decoded losslessly.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .base import Tokenizer
from .pretokenize import count_pretokens

# A byte-fallback character scores this much below the least likely piece
# (SentencePiece's kUnkPenalty).
_UNK_PENALTY = 10.0

Pieces = dict[str, float]  # piece -> log probability


def _lattice_params(pieces: Pieces) -> tuple[int, float]:
    """(longest piece length, score of a byte-fallback character)."""
    return max(map(len, pieces)), min(pieces.values()) - _UNK_PENALTY


def viterbi(
    word: str,
    pieces: Pieces,
    exclude: str | None = None,
    params: tuple[int, float] | None = None,
) -> tuple[float, list[str]]:
    """Best segmentation of word: the pieces (in order) with the highest
    total log probability, and that total.

    A character that is not itself a piece is a byte-fallback unit with the
    penalty score. exclude names one piece that may not be used (to find a
    piece's best alternative segmentation while pruning). Among equal
    scores the segmentation whose last piece starts earliest wins.

    params is _lattice_params(pieces), passed in when segmenting many words
    with the same pieces (computing it scans every piece).
    """
    max_len, unk_score = params or _lattice_params(pieces)
    n = len(word)
    best = [-math.inf] * (n + 1)
    best[0] = 0.0
    back = [0] * (n + 1)
    for end in range(1, n + 1):
        for start in range(max(0, end - max_len), end):
            if best[start] == -math.inf:
                continue
            sub = word[start:end]
            lp = pieces.get(sub)
            if lp is None or sub == exclude:
                if end - start == 1 and sub not in pieces:
                    lp = unk_score
                else:
                    continue
            score = best[start] + lp
            if score > best[end]:
                best[end] = score
                back[end] = start
    segments = []
    end = n
    while end > 0:
        segments.append(word[back[end] : end])
        end = back[end]
    return best[n], segments[::-1]


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def forward_backward(
    word: str, pieces: Pieces, params: tuple[int, float] | None = None
) -> tuple[float, dict[str, float]]:
    """Marginal log likelihood of word (summed over all segmentations) and
    the expected number of times each piece occurs in a segmentation drawn
    from the model. Byte-fallback characters contribute to the likelihood
    but not to the counts. params as for viterbi."""
    max_len, unk_score = params or _lattice_params(pieces)
    n = len(word)
    # edges[start] = [(end, piece or None, log prob)]
    edges: list[list[tuple[int, str | None, float]]] = [[] for _ in range(n)]
    for start in range(n):
        for end in range(start + 1, min(n, start + max_len) + 1):
            sub = word[start:end]
            lp = pieces.get(sub)
            if lp is not None:
                edges[start].append((end, sub, lp))
            elif end - start == 1:
                edges[start].append((end, None, unk_score))

    incoming: list[list[float]] = [[] for _ in range(n + 1)]
    alpha = [-math.inf] * (n + 1)
    alpha[0] = 0.0
    for start in range(n):
        if start:
            alpha[start] = _logsumexp(incoming[start])
        for end, _, lp in edges[start]:
            incoming[end].append(alpha[start] + lp)
    alpha[n] = _logsumexp(incoming[n])

    beta = [-math.inf] * (n + 1)
    beta[n] = 0.0
    for start in range(n - 1, -1, -1):
        beta[start] = _logsumexp([lp + beta[end] for end, _, lp in edges[start]])

    log_z = alpha[n]
    counts: dict[str, float] = {}
    for start in range(n):
        for end, piece, lp in edges[start]:
            if piece is not None:
                p = math.exp(alpha[start] + lp + beta[end] - log_z)
                counts[piece] = counts.get(piece, 0.0) + p
    return log_z, counts


def e_step(
    words: list[tuple[str, int]], pieces: Pieces
) -> tuple[Counter[str], float]:
    """Expected piece counts over a corpus of (word, frequency), and the
    corpus log likelihood."""
    counts: Counter[str] = Counter()
    loglik = 0.0
    params = _lattice_params(pieces)
    for word, freq in words:
        log_z, word_counts = forward_backward(word, pieces, params)
        loglik += freq * log_z
        for piece, c in word_counts.items():
            counts[piece] += freq * c
    return counts, loglik


def m_step(
    counts: Counter[str], required: Iterable[str], min_count: float = 0.5
) -> Pieces:
    """Maximum-likelihood probabilities from expected counts. Pieces with
    an expected count below min_count are dropped, except the required
    characters, which always stay (with at least min_count)."""
    kept = {p: c for p, c in counts.items() if c > 0 and c >= min_count}
    for ch in required:
        kept[ch] = max(counts.get(ch, 0.0), min_count) or 1e-6
    log_total = math.log(sum(kept.values()))
    return {p: math.log(c) - log_total for p, c in kept.items()}


def prune(words: list[tuple[str, int]], pieces: Pieces, size: int) -> Pieces:
    """Shrink pieces to size, dropping first the pieces whose removal is
    estimated to lose the least likelihood (SentencePiece's approximation).

    For each multi-char piece, its best alternative segmentation (without
    it) is found. Its loss is how much the Viterbi-segmented corpus's log
    likelihood drops when its occurrences are re-segmented that way, weighted
    by the share of the corpus whose segmentation uses it. A piece that is
    not even its own best segmentation, or that no word uses, costs nothing.
    Single characters are always kept.
    """
    params = _lattice_params(pieces)
    alternatives: dict[str, list[str]] = {}
    free: set[str] = set()
    for p in pieces:
        if len(p) == 1:
            continue
        if viterbi(p, pieces, params=params)[1] != [p]:
            free.add(p)
        else:
            alternatives[p] = viterbi(p, pieces, exclude=p, params=params)[1]

    freq: Counter[str] = Counter()  # occurrences in Viterbi segmentations
    word_share: Counter[str] = Counter()  # frequency of words using the piece
    for word, f in words:
        segments = viterbi(word, pieces, params=params)[1]
        for s in segments:
            freq[s] += f
        for s in set(segments):
            word_share[s] += f
    vsum = sum(freq.values())

    kept: Pieces = {p: lp for p, lp in pieces.items() if len(p) == 1}
    candidates: list[tuple[float, str]] = []
    for p in pieces:
        if len(p) == 1:
            continue
        if freq[p] == 0 or p in free:
            candidates.append((0.0, p))
            continue
        alt = alternatives[p]
        logprob_sp = math.log(freq[p]) - math.log(vsum)
        # Removing p turns each of its occurrences into the len(alt) pieces
        # of alt, whose counts (and the total) grow accordingly.
        logsum_alt = math.log(vsum + freq[p] * (len(alt) - 1))
        logprob_alt = sum(math.log(freq[a] + freq[p]) - logsum_alt for a in alt)
        loss = word_share[p] / vsum * (logprob_sp - logprob_alt)
        candidates.append((loss, p))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    for _, p in candidates:
        if len(kept) >= size:
            break
        kept[p] = pieces[p]
    return kept


def required_chars(word_freqs: Counter[str], coverage: float) -> list[str]:
    """The most frequent characters that together make up at least
    `coverage` of all character occurrences (the rest use byte fallback)."""
    char_freq: Counter[str] = Counter()
    for word, f in word_freqs.items():
        for ch in word:
            char_freq[ch] += f
    total = sum(char_freq.values())
    chars, covered = [], 0
    for ch, f in sorted(char_freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if covered >= coverage * total:
            break
        chars.append(ch)
        covered += f
    return chars


def seed_pieces(
    word_freqs: Counter[str], required: list[str], max_len: int, seed_size: int
) -> Pieces:
    """Initial vocabulary: the required characters plus the seed_size
    substrings (2..max_len chars, required characters only) with the
    highest frequency x length, with log probabilities proportional to
    those scores."""
    allowed = set(required)
    sub_freq: Counter[str] = Counter()
    char_freq: Counter[str] = Counter()
    for word, f in word_freqs.items():
        n = len(word)
        for i in range(n):
            if word[i] not in allowed:
                continue
            char_freq[word[i]] += f
            for j in range(i + 2, min(n, i + max_len) + 1):
                if word[j - 1] not in allowed:
                    break
                sub_freq[word[i:j]] += f
    scored = sorted(
        ((f * len(s), s) for s, f in sub_freq.items()), key=lambda x: (-x[0], x[1])
    )[:seed_size]
    scores = {s: float(score) for score, s in scored}
    for ch in required:
        scores[ch] = float(char_freq[ch])
    log_total = math.log(sum(scores.values()))
    return {p: math.log(s) - log_total for p, s in scores.items()}


class UnigramTokenizer(Tokenizer):
    """Unigram LM tokenizer with byte fallback.

    Ids: 0..255 are raw bytes (fallback), then the pieces in order of
    decreasing probability, then SPECIAL_TOKENS.
    """

    _CACHE_LIMIT = 1 << 20

    def __init__(self) -> None:
        super().__init__()
        self.pieces: Pieces = {}
        self.vocab: dict[str, int] = {}  # piece -> id
        self.inverse_vocab: dict[int, str] = {}
        self._cache: dict[str, list[int]] = {}

    def train(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int,
        verbose: bool = False,
        num_workers: int = 1,
        piece_chars: int = 1 << 20,
        character_coverage: float = 0.9995,
        max_piece_len: int = 16,
        seed_size: int = 1_000_000,
        shrinking_factor: float = 0.75,
        num_sub_iterations: int = 2,
    ) -> None:
        """Learn pieces from documents (texts as for count_pretokens).

        vocab_size counts everything: 256 byte ids + pieces + SPECIAL_TOKENS.
        It is an upper bound: a corpus with fewer candidate pieces gives a
        smaller vocab (len(self) is the actual size). Pruning stops once at most 1.1x the piece budget remains, then the
        most probable pieces (and all required characters) are kept.
        """
        budget = vocab_size - 256 - len(self.SPECIAL_TOKENS)
        if budget < 1:
            raise ValueError(f"vocab_size must be > {256 + len(self.SPECIAL_TOKENS)}")
        word_freqs = count_pretokens(texts, num_workers, piece_chars)
        required = required_chars(word_freqs, character_coverage)
        if len(required) > budget:
            raise ValueError(
                f"{len(required)} characters are needed for coverage "
                f"{character_coverage}, but only {budget} pieces fit"
            )
        pieces = seed_pieces(word_freqs, required, max_piece_len, seed_size)
        words = list(word_freqs.items())
        del word_freqs

        desired = int(budget * 1.1)
        while True:
            for _ in range(num_sub_iterations):
                counts, loglik = e_step(words, pieces)
                pieces = m_step(counts, required)
                if verbose:
                    print(f"EM: {len(pieces)} pieces, log likelihood {loglik:.1f}")
            if len(pieces) <= desired:
                break
            pieces = prune(words, pieces, max(desired, int(len(pieces) * shrinking_factor)))
            if verbose:
                print(f"pruned to {len(pieces)} pieces")

        keep = set(required)
        for p, _ in sorted(pieces.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(keep) >= budget:
                break
            keep.add(p)
        self._build({p: pieces[p] for p in keep})

    def _build(self, pieces: Pieces, special_tokens: Iterable[str] | None = None) -> None:
        self.pieces = pieces
        order = sorted(pieces.items(), key=lambda kv: (-kv[1], kv[0]))
        self.vocab = {p: 256 + i for i, (p, _) in enumerate(order)}
        first_special = 256 + len(order)
        for token in self.SPECIAL_TOKENS:
            # Pretokenization splits "<|", the name and "|>" apart, so no
            # piece can be a whole special token.
            assert token not in self.vocab, token
        specials = self.SPECIAL_TOKENS if special_tokens is None else special_tokens
        self._set_special_tokens({t: first_special + i for i, t in enumerate(specials)})
        self.inverse_vocab = {i: p for p, i in self.vocab.items()}
        self._params = _lattice_params(pieces)
        self._cache = {}

    def __len__(self) -> int:
        return 256 + len(self.vocab) + len(self.special_tokens)

    def _encode_word(self, word: str) -> list[int]:
        ids = []
        for seg in viterbi(word, self.pieces, params=self._params)[1]:
            i = self.vocab.get(seg)
            if i is None:  # byte fallback
                ids.extend(seg.encode("utf-8"))
            else:
                ids.append(i)
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
        return self.inverse_vocab[token_id].encode("utf-8")

    def save(self, path: str | Path) -> None:
        data = {
            "type": "unigram",
            "pieces": [[p, lp] for p, lp in self.pieces.items()],
            "special_tokens": list(self.special_tokens),
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "UnigramTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok._build({p: lp for p, lp in data["pieces"]}, data["special_tokens"])
        return tok
