"""Pre-tokenization shared by all tokenizers: split text into word-like
pieces at script boundaries, and count those pieces over a corpus (optionally
in worker processes)."""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait

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


def pretokenize(text: str) -> list[str]:
    return _SPLIT_PATTERN.findall(text)


def pretokenize_chunks(chunks: Iterable[str]) -> Iterator[str]:
    """Pretokenize one document given as consecutive text chunks.

    Yields exactly what pretokenize("".join(chunks)) would, without ever
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


def count_pretokens(
    texts: Iterable[str | Iterable[str]],
    num_workers: int = 1,
    piece_chars: int = 1 << 20,
) -> Counter[str]:
    """Count pretokens over a corpus.

    Each document is either a str or an iterable of str chunks (e.g. a
    large file read piece by piece), and texts itself may be a generator,
    so the corpus is streamed: only the counts of distinct pretokens are
    kept in memory, never the corpus text.

    With num_workers > 1, pretokenizing and counting run in worker
    processes on pieces of about piece_chars characters, with the same
    result as a single process.
    """
    docs = ((doc,) if isinstance(doc, str) else doc for doc in texts)
    if num_workers > 1:
        return _count_parallel(docs, num_workers, piece_chars)
    token_freqs: Counter[str] = Counter()
    for chunks in docs:
        token_freqs.update(pretokenize_chunks(chunks))
    return token_freqs


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
