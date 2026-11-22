"""Answering with a fine-tuned model, given the paragraphs."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

import torch

from webbot.model import GPT
from webbot.tokenizer.base import Tokenizer

from .data import Doc
from .format import ANSWER, encode_docs, encode_prompt, encode_search


@torch.no_grad()
def generate_answer(
    model: GPT,
    tokenizer: Tokenizer,
    question: str,
    docs: Sequence[Doc],
    query: str | None = None,
    max_new_tokens: int = 48,
) -> str:
    """Greedy answer after the question, the search (query, default the
    question) and the paragraphs, up to <|endoftext|>."""
    device = next(model.parameters()).device
    ids = (encode_prompt(tokenizer, question) + encode_search(tokenizer, query or question)
           + encode_docs(tokenizer, docs) + [tokenizer.special_tokens[ANSWER]])
    ids = ids[-(model.config.max_seq_len - max_new_tokens):]
    eot = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    was_training = model.training
    model.eval()
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens, temperature=0, eot_id=eot)
    model.train(was_training)
    new = [t for t in out[0, len(ids):].tolist()]
    if eot in new:
        new = new[: new.index(eot)]
    return tokenizer.decode(new)


def normalize_answer(text: str) -> str:
    """NFKC, no whitespace: so 「Ｗａｔｔａ! Itta!」 matches 「Watta!Itta!」."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def exact_match(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)
