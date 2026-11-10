"""Behaviour shared by all tokenizers: special tokens, encode() with
allowed special tokens, streaming / multi-document encoding and decoding.

A subclass provides the vocabulary-specific parts: _encode_pretokens (ids
for a batch of pretokens) and _token_bytes (the bytes an ordinary id stands
for), and assigns special-token ids with _add_special_tokens."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from .pretokenize import pretokenize, pretokenize_chunks


class Tokenizer:
    # Document-boundary marker for pretraining data, also ends an answer.
    EOT_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"

    # Special tokens get the ids right after the ordinary vocabulary, in
    # this order. Training never produces them; encode() emits them only
    # when explicitly allowed, so the same strings inside ordinary text
    # (e.g. a Wikipedia article) are encoded as plain text. Retrieval-QA:
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

    # Pretokens encoded per batch in encode_chunks.
    _ENCODE_BATCH = 4096

    def __init__(self) -> None:
        self.special_tokens: dict[str, int] = {}
        self._special_ids: dict[int, str] = {}

    # --- implemented by subclasses -------------------------------------

    def __len__(self) -> int:
        """Number of ids (bytes / pieces + special tokens): the model's
        embedding size."""
        raise NotImplementedError

    def _encode_pretokens(self, pretokens: list[str]) -> list[int]:
        raise NotImplementedError

    def _token_bytes(self, token_id: int) -> bytes:
        raise NotImplementedError

    # --- shared ----------------------------------------------------------

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        return pretokenize(text)

    @staticmethod
    def _pretokenize_chunks(chunks: Iterable[str]) -> Iterator[str]:
        return pretokenize_chunks(chunks)

    def _set_special_tokens(self, ids: dict[str, int]) -> None:
        self.special_tokens = dict(ids)
        self._special_ids = {i: t for t, i in self.special_tokens.items()}

    def _allowed_set(self, allowed_special: Iterable[str] | str) -> set[str]:
        if allowed_special == "all":
            return set(self.special_tokens)
        allowed = set(allowed_special)
        unknown = allowed - self.special_tokens.keys()
        if unknown:
            raise ValueError(f"not special tokens: {sorted(unknown)}")
        return allowed

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
        return self._encode_pretokens(self._pretokenize(text))

    def encode_chunks(self, chunks: Iterable[str]) -> Iterator[int]:
        """Lazily encode one document given as consecutive text chunks.

        Same ids as encode("".join(chunks)), without materializing the text
        or the full id list; callers can write ids out as they arrive.
        """
        batch: list[str] = []
        for token in self._pretokenize_chunks(chunks):
            batch.append(token)
            if len(batch) >= self._ENCODE_BATCH:
                yield from self._encode_pretokens(batch)
                batch = []
        yield from self._encode_pretokens(batch)

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
                byte_seq.extend(self._token_bytes(i))
        return byte_seq.decode("utf-8", errors="replace")
