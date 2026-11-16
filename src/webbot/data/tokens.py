from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from webbot.tokenizer.base import Tokenizer


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def write_token_file(
    tokenizer: Tokenizer,
    docs: Iterable[str | Iterable[str]],
    path: str | Path,
    buffer_tokens: int = 1 << 20,
) -> int:
    """Encode documents into one flat token file for pretraining.

    Every document is followed by <|endoftext|>. Documents may be strings or
    iterables of text chunks, and ids are flushed to disk every
    buffer_tokens, so neither the corpus nor the id stream is ever held in
    memory. Ids are stored as uint16 when the vocab fits (half the size of
    uint32); the dtype and token count go to a "<path>.json" sidecar.
    Returns the number of tokens written.
    """
    path = Path(path)
    dtype = np.uint16 if len(tokenizer) <= 1 << 16 else np.uint32
    eot_id = tokenizer.special_tokens[tokenizer.EOT_TOKEN]
    n_tokens = 0
    buf: list[int] = []
    with path.open("wb") as f:

        def flush() -> None:
            nonlocal n_tokens
            np.asarray(buf, dtype=dtype).tofile(f)
            n_tokens += len(buf)
            buf.clear()

        for doc in docs:
            chunks = (doc,) if isinstance(doc, str) else doc
            for token_id in tokenizer.encode_chunks(chunks):
                buf.append(token_id)
                if len(buf) >= buffer_tokens:
                    flush()
            buf.append(eot_id)
        flush()
    _meta_path(path).write_text(
        json.dumps({"dtype": np.dtype(dtype).name, "num_tokens": n_tokens}),
        encoding="utf-8",
    )
    return n_tokens


def load_token_file(path: str | Path) -> np.memmap:
    """Memory-map a file written by write_token_file (read-only, lazy)."""
    path = Path(path)
    meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
    return np.memmap(path, dtype=meta["dtype"], mode="r", shape=(meta["num_tokens"],))


def split_train_val(
    tokens: np.ndarray, val_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last val_fraction of the token stream for validation.

    A contiguous tail (rather than random windows) guarantees that no
    validation token is ever seen in training. Both parts are views.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    cut = int(len(tokens) * (1 - val_fraction))
    return tokens[:cut], tokens[cut:]


def _to_batch(windows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.from_numpy(windows.astype(np.int64))
    return w[:, :-1], w[:, 1:]


class RandomBatches:
    """Endless (input, target) batches from random windows of a token array.

    The target is the input shifted by one token. The numpy RNG state is
    exposed so a resumed run draws exactly the batches it would have drawn.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int = 0):
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"need at least seq_len+1={seq_len + 1} tokens, got {len(tokens)}"
            )
        self.tokens = tokens
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len, size=self.batch_size)
        offsets = np.arange(self.seq_len + 1)
        return _to_batch(np.asarray(self.tokens[starts[:, None] + offsets]))

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]


def sequential_batches(
    tokens: np.ndarray, seq_len: int, batch_size: int, max_batches: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Non-overlapping windows from the start of tokens, for a deterministic
    evaluation set that is identical at every evaluation."""
    n_windows = min((len(tokens) - 1) // seq_len, batch_size * max_batches)
    for start in range(0, n_windows, batch_size):
        idx = np.arange(start, min(start + batch_size, n_windows))
        offsets = np.arange(seq_len + 1)
        yield _to_batch(np.asarray(tokens[idx[:, None] * seq_len + offsets]))
