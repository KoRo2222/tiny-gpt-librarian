import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from webbot.data import (
    RandomBatches,
    load_token_file,
    sequential_batches,
    split_train_val,
    write_token_file,
)
from webbot.tokenizer import BPETokenizer

DOCS = [
    "むかしむかし、ある国に王様がいました。",
    "王様は毎晩、物語を聞きました。",
    "ドラゴンは城にやってきました。",
]


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train(DOCS, vocab_size=300)
    return tok


def test_token_file_is_docs_separated_by_eot(tokenizer, tmp_path):
    path = tmp_path / "tokens.bin"
    n = write_token_file(tokenizer, DOCS, path)
    tokens = load_token_file(path)
    assert tokens.dtype == np.uint16
    assert n == len(tokens)
    assert tokens.tolist() == tokenizer.encode_with_eot(DOCS)


def test_token_file_streams_chunked_docs_and_small_buffers(tokenizer, tmp_path):
    # Documents as chunk iterators and a tiny flush buffer must give the
    # very same file as whole strings.
    whole, chunked = tmp_path / "a.bin", tmp_path / "b.bin"
    write_token_file(tokenizer, DOCS, whole)
    write_token_file(
        tokenizer,
        (iter([d[i : i + 3] for i in range(0, len(d), 3)]) for d in DOCS),
        chunked,
        buffer_tokens=4,
    )
    assert whole.read_bytes() == chunked.read_bytes()


def test_split_holds_out_disjoint_tail():
    tokens = np.arange(100, dtype=np.uint16)
    train, val = split_train_val(tokens, 0.1)
    assert train.tolist() == list(range(90))
    assert val.tolist() == list(range(90, 100))
    with pytest.raises(ValueError):
        split_train_val(tokens, 0.0)


def test_random_batch_target_is_input_shifted_by_one():
    tokens = np.arange(1000, dtype=np.uint16)  # token value == its position
    x, y = RandomBatches(tokens, seq_len=16, batch_size=8, seed=0).next()
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.long
    assert torch.equal(y, x + 1)
    # Windows are contiguous runs of the stream.
    assert torch.equal(x[:, 1:] - x[:, :-1], torch.ones(8, 15, dtype=torch.long))


def test_random_batches_cover_the_whole_range_and_stay_inside():
    tokens = np.arange(50, dtype=np.uint16)
    batches = RandomBatches(tokens, seq_len=10, batch_size=64, seed=0)
    starts = set()
    for _ in range(20):
        x, y = batches.next()
        assert y.max() <= 49
        starts.update(x[:, 0].tolist())
    assert starts == set(range(40))  # every valid start gets sampled


def test_random_batches_resume_from_state():
    tokens = np.arange(1000, dtype=np.uint16)
    a = RandomBatches(tokens, 8, 4, seed=1)
    a.next()
    state = a.state_dict()
    expected = [a.next()[0] for _ in range(3)]
    b = RandomBatches(tokens, 8, 4, seed=999)
    b.load_state_dict(state)
    assert all(torch.equal(e, b.next()[0]) for e in expected)


def test_sequential_batches_are_fixed_non_overlapping_windows():
    tokens = np.arange(100, dtype=np.uint16)
    batches = list(sequential_batches(tokens, seq_len=10, batch_size=4, max_batches=2))
    starts = torch.cat([x[:, 0] for x, _ in batches]).tolist()
    assert starts == [0, 10, 20, 30, 40, 50, 60, 70]
    for x, y in batches:
        assert torch.equal(y, x + 1)
    # Limited by data: only 9 full windows fit in 100 tokens.
    all_windows = list(sequential_batches(tokens, 10, 4, max_batches=100))
    assert sum(len(x) for x, _ in all_windows) == 9
