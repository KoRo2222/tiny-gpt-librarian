"""BM25 retrieval over an inverted index (Robertson & Zaragoza; the Lucene
form of the formula).

    score(q, d) = sum over distinct terms t of q of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

tf(t, d) is how often t occurs in document d, |d| the document's length in
terms, avgdl the mean length, N the number of documents and df(t) how many
contain t. k1 sets how fast repeated occurrences saturate, b how strongly
long documents are discounted. This idf never goes negative.

Japanese has no spaces between words, so the analyzer that turns text into
terms matters: char_bigrams (dictionary-free, the usual choice for CJK full
text search) or any other function from text to a list of terms.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

Analyzer = Callable[[str], list]

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC (full-width ASCII and half-width kana to their usual forms)
    and lower case."""
    return unicodedata.normalize("NFKC", text).lower()


def char_bigrams(text: str) -> list[str]:
    """Overlapping 2-character terms of each whitespace-separated run of the
    normalized text; a run of one character is a term by itself."""
    terms = []
    for run in _SPACE.split(normalize(text)):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


class BM25Index:
    """An inverted index of documents with BM25 search.

    Postings are stored per term as parallel arrays (document ids, term
    frequencies) in one CSR layout: term i's postings are
    doc_ids[offsets[i]:offsets[i + 1]].
    """

    def __init__(self, analyzer: Analyzer = char_bigrams, k1: float = 1.2, b: float = 0.75):
        self.analyzer = analyzer
        self.k1, self.b = k1, b
        self.vocab: dict = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int32)
        self.tfs = np.zeros(0, dtype=np.int32)
        self.doc_len = np.zeros(0, dtype=np.int32)

    @property
    def num_docs(self) -> int:
        return len(self.doc_len)

    def build(self, docs: Iterable[str]) -> "BM25Index":
        """Index the documents; their ids are their positions (0, 1, ...)."""
        vocab: dict = {}
        term_ids: list[int] = []
        doc_of: list[int] = []
        counts: list[int] = []
        lengths: list[int] = []
        for d, text in enumerate(docs):
            terms = self.analyzer(text)
            lengths.append(len(terms))
            for term, c in Counter(terms).items():
                term_ids.append(vocab.setdefault(term, len(vocab)))
                doc_of.append(d)
                counts.append(c)
        term_arr = np.asarray(term_ids, dtype=np.int64)
        # Group postings by term (stable: documents stay in increasing order).
        order = np.argsort(term_arr, kind="stable")
        self.vocab = vocab
        self.doc_ids = np.asarray(doc_of, dtype=np.int32)[order]
        self.tfs = np.asarray(counts, dtype=np.int32)[order]
        self.offsets = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(np.bincount(term_arr, minlength=len(vocab)), out=self.offsets[1:])
        self.doc_len = np.asarray(lengths, dtype=np.int32)
        return self

    def idf(self, df: np.ndarray | int) -> np.ndarray:
        n = self.num_docs
        return np.log1p((n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for the query (float64)."""
        out = np.zeros(self.num_docs, dtype=np.float64)
        if self.num_docs == 0:
            return out
        avgdl = self.doc_len.mean()
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / avgdl)
        for term in dict.fromkeys(self.analyzer(query)):  # distinct, in order
            t = self.vocab.get(term)
            if t is None:
                continue
            lo, hi = self.offsets[t], self.offsets[t + 1]
            docs, tf = self.doc_ids[lo:hi], self.tfs[lo:hi]
            # A term's postings hold each document once, so += is safe.
            out[docs] += self.idf(hi - lo) * tf * (self.k1 + 1) / (tf + norm[docs])
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """The k best (doc id, score) pairs with a positive score, best
        first; equal scores go to the lower doc id."""
        s = self.scores(query)
        k = min(k, int(np.count_nonzero(s > 0)))
        if k == 0:
            return []
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        # Candidates tied with the k-th score may sit outside the partition:
        # take every document scoring at least that much, then sort.
        cutoff = s[top].min()
        top = np.flatnonzero(s >= cutoff)
        top = top[np.lexsort((top, -s[top]))][:k]
        return [(int(d), float(s[d])) for d in top]

    def save(self, path: str | Path) -> None:
        """Write the index (not the analyzer) to path.npz + path.vocab.json."""
        path = Path(path)
        terms = sorted(self.vocab, key=self.vocab.get)
        np.savez(path.with_suffix(".npz"), offsets=self.offsets, doc_ids=self.doc_ids,
                 tfs=self.tfs, doc_len=self.doc_len, params=np.array([self.k1, self.b]))
        path.with_suffix(".vocab.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, analyzer: Analyzer = char_bigrams) -> "BM25Index":
        """Read an index written by save; analyzer must be the one it was built with."""
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        k1, b = data["params"].tolist()
        index = cls(analyzer, k1, b)
        terms = json.loads(path.with_suffix(".vocab.json").read_text(encoding="utf-8"))
        index.vocab = {_term(t): i for i, t in enumerate(terms)}
        index.offsets, index.doc_ids = data["offsets"], data["doc_ids"]
        index.tfs, index.doc_len = data["tfs"], data["doc_len"]
        return index


def _term(t):
    # JSON turns tuple terms into lists; terms must be hashable.
    return tuple(t) if isinstance(t, list) else t


def token_analyzer(tokenizer) -> Analyzer:
    """Terms = the token ids of a webbot tokenizer (on normalized text)."""
    return lambda text: tokenizer.encode(normalize(text))


__all__: Sequence[str] = ["BM25Index", "char_bigrams", "normalize", "token_analyzer"]
