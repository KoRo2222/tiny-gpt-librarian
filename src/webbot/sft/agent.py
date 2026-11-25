"""The full loop: the model writes a search query, the BM25 index is
searched with it, and the model answers from what came back.

    <|question|>Q<|search|>  -> the model writes the query up to <|/search|>
    + the top-k paragraphs as <|doc|>...<|/doc|>, then <|answer|>
                             -> the model writes the answer up to <|endoftext|>

If the model writes no usable query (it stops at once, or runs into
another special token), the question itself is searched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from webbot.model import GPT
from webbot.retrieval import ParagraphIndex
from webbot.tokenizer.base import Tokenizer

from .data import Doc, make_doc
from .format import END_SEARCH, SEARCH, encode_prompt
from .infer import generate_answer


@dataclass(frozen=True)
class Reply:
    query: str
    docs: tuple[Doc, ...]
    answer: str


@torch.no_grad()
def write_query(model: GPT, tokenizer: Tokenizer, question: str, max_new_tokens: int = 64) -> str:
    """The search query the model writes after the question ("" if none)."""
    device = next(model.parameters()).device
    ids = encode_prompt(tokenizer, question) + [tokenizer.special_tokens[SEARCH]]
    end = tokenizer.special_tokens[END_SEARCH]
    was_training = model.training
    model.eval()
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens, temperature=0, eot_id=end)
    model.train(was_training)
    new = out[0, len(ids):].tolist()
    specials = set(tokenizer.special_tokens.values())
    query = []
    for t in new:
        if t in specials:  # <|/search|> ends the query; any other marker is a malformed one
            break
        query.append(t)
    return tokenizer.decode(query).strip()


def ask(
    model: GPT,
    tokenizer: Tokenizer,
    index: ParagraphIndex,
    question: str,
    k: int = 3,
    max_doc_chars: int = 400,
    use_model_query: bool = True,
) -> Reply:
    """Search (with the model's own query, or the question when
    use_model_query is False or the model wrote none) and answer."""
    query = write_query(model, tokenizer, question) if use_model_query else ""
    query = query or question
    docs = tuple(make_doc(p, max_doc_chars) for p, _ in index.search(query, k))
    return Reply(query, docs, generate_answer(model, tokenizer, question, docs, query))


def evaluate_end_to_end(
    model: GPT,
    tokenizer: Tokenizer,
    index: ParagraphIndex,
    qas,
    k: int = 3,
    use_model_query: bool = True,
    log=None,
) -> dict[str, float]:
    """Over questions (retrieval.QA), the fractions where the answer is
    exactly right ("exact_match"), contained in the reply ("contains_answer"),
    present in the retrieved paragraphs ("answer_in_docs"), and where the
    model's query was the question verbatim ("query_is_question")."""
    from .infer import exact_match, normalize_answer

    counts = {"exact_match": 0, "contains_answer": 0, "answer_in_docs": 0, "query_is_question": 0}
    for i, qa in enumerate(qas):
        reply = ask(model, tokenizer, index, qa.query, k, use_model_query=use_model_query)
        counts["exact_match"] += exact_match(reply.answer, qa.answer)
        counts["contains_answer"] += normalize_answer(qa.answer) in normalize_answer(reply.answer)
        counts["answer_in_docs"] += any(qa.answer in d.text for d in reply.docs)
        counts["query_is_question"] += reply.query == qa.query
        if log is not None and i < 5:
            log(f"Q: {qa.query} | query: {reply.query} | gold: {qa.answer} | pred: {reply.answer}")
    return {name: c / len(qas) for name, c in counts.items()} if qas else {}
