"""The token layout of a search-read-answer exchange:

    <|question|>Q <|search|>query<|/search|> (<|doc|>title / section\\ntext<|/doc|>)* <|answer|>A <|endoftext|>

The model writes the search (from <|search|> through <|/search|>) and the
answer (from <|answer|> through <|endoftext|>); the question and the
paragraphs come from outside. The loss mask is True exactly on the tokens
the model writes.

Markers are inserted as their special token ids and every piece of text
is encoded as ordinary text, so a marker-like string inside a Wikipedia
paragraph or a question stays plain text. Training and inference encode
the same pieces the same way (encode_prompt, encode_docs, ...), so the
token boundaries agree.
"""

from __future__ import annotations

from collections.abc import Sequence

from webbot.tokenizer.base import Tokenizer

from .data import Doc, SFTExample

QUESTION, SEARCH, END_SEARCH = "<|question|>", "<|search|>", "<|/search|>"
DOC, END_DOC, ANSWER = "<|doc|>", "<|/doc|>", "<|answer|>"


def doc_text(d: Doc) -> str:
    header = d.title + (f" / {d.section}" if d.section else "")
    return f"{header}\n{d.text}"


def _special(tokenizer: Tokenizer, token: str) -> int:
    return tokenizer.special_tokens[token]


def encode_prompt(tokenizer: Tokenizer, question: str) -> list[int]:
    return [_special(tokenizer, QUESTION)] + tokenizer.encode(question)


def encode_search(tokenizer: Tokenizer, query: str) -> list[int]:
    return [_special(tokenizer, SEARCH)] + tokenizer.encode(query) + [_special(tokenizer, END_SEARCH)]


def encode_docs(tokenizer: Tokenizer, docs: Sequence[Doc]) -> list[int]:
    ids: list[int] = []
    for d in docs:
        ids += [_special(tokenizer, DOC)] + tokenizer.encode(doc_text(d)) + [_special(tokenizer, END_DOC)]
    return ids


def encode_answer(tokenizer: Tokenizer, answer: str) -> list[int]:
    return (
        [_special(tokenizer, ANSWER)]
        + tokenizer.encode(answer)
        + [_special(tokenizer, tokenizer.EOT_TOKEN)]
    )


def encode_example(tokenizer: Tokenizer, ex: SFTExample) -> tuple[list[int], list[bool]]:
    """Token ids of the whole exchange and, per token, whether the model
    writes it (and is trained to predict it)."""
    ids: list[int] = []
    mask: list[bool] = []
    for piece, learned in (
        (encode_prompt(tokenizer, ex.question), False),
        (encode_search(tokenizer, ex.query), True),
        (encode_docs(tokenizer, ex.docs), False),
        (encode_answer(tokenizer, ex.answer), True),
    ):
        ids += piece
        mask += [learned] * len(piece)
    return ids, mask
