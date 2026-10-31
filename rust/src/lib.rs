//! Native BPE training and encoding for webbot's byte-level BPE tokenizer.
//!
//! Symbols are u32 ids: 0..=255 are the raw bytes, and the i-th learned merge
//! creates id 256 + i, matching the vocab layout of the Python tokenizer.
//! Both functions reproduce the Python reference implementation exactly
//! (same merges, same ids); the Python code is the specification.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

type Pair = (u32, u32);

/// Heap entry: highest count first, ties broken toward the lexicographically
/// larger pair of symbol strings (Python's `max(key=(count, pair))`).
struct Entry {
    count: i64,
    pair: Pair,
    a: Arc<str>,
    b: Arc<str>,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| (&*self.a, &*self.b).cmp(&(&*other.a, &*other.b)))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

/// Merge every non-overlapping occurrence of `pair` (left to right) in `word`
/// into `new_id`, calling `delta(pair, ±1)` for each adjacent pair whose
/// occurrence count changes. Only the neighbours of a merge site change, so
/// the cost is proportional to the word length, with O(1) count updates per
/// merge site instead of re-counting every pair of the word.
/// Returns None if the word does not contain the pair.
fn merge_word(
    word: &[u32],
    pair: Pair,
    new_id: u32,
    mut delta: impl FnMut(Pair, i64),
) -> Option<Vec<u32>> {
    let (a, b) = pair;
    let mut out: Vec<u32> = Vec::with_capacity(word.len());
    let mut found = false;
    let mut i = 0;
    while i < word.len() {
        if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
            found = true;
            delta(pair, -1);
            // Left neighbour: taken from the output, so a preceding merge
            // site in this same word is already the new symbol.
            if let Some(&prev) = out.last() {
                delta((prev, a), -1);
                delta((prev, new_id), 1);
            }
            // Right neighbour: still the original symbol; if it starts
            // another merge site, that site's left-neighbour update above
            // undoes this one.
            if i + 2 < word.len() {
                let next = word[i + 2];
                delta((b, next), -1);
                delta((new_id, next), 1);
            }
            out.push(new_id);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    found.then_some(out)
}

/// Learn up to `num_merges` merges from (pretoken, count) pairs.
///
/// `base_symbols[b]` is the printable string of byte b; symbol strings are
/// only used to break count ties the same way the Python trainer does.
/// Returns (left, right, count) for each merge in order.
#[pyfunction]
fn train_bpe(
    py: Python<'_>,
    words: Vec<(String, i64)>,
    base_symbols: Vec<String>,
    num_merges: usize,
) -> PyResult<Vec<(String, String, i64)>> {
    if base_symbols.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "base_symbols must have 256 entries",
        ));
    }
    py.detach(|| Ok(train_impl(words, base_symbols, num_merges)))
}

fn train_impl(
    words: Vec<(String, i64)>,
    base_symbols: Vec<String>,
    num_merges: usize,
) -> Vec<(String, String, i64)> {
    let mut symbols: Vec<Arc<str>> = base_symbols.into_iter().map(Arc::from).collect();
    let mut freqs: Vec<i64> = Vec::with_capacity(words.len());
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(words.len());
    for (token, freq) in words {
        seqs.push(token.bytes().map(u32::from).collect());
        freqs.push(freq);
    }

    let mut pair_counts: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut pair_to_words: FxHashMap<Pair, FxHashSet<u32>> = FxHashMap::default();
    for (idx, seq) in seqs.iter().enumerate() {
        for w in seq.windows(2) {
            let p = (w[0], w[1]);
            *pair_counts.entry(p).or_insert(0) += freqs[idx];
            pair_to_words.entry(p).or_default().insert(idx as u32);
        }
    }

    let entry = |symbols: &[Arc<str>], pair: Pair, count: i64| Entry {
        count,
        pair,
        a: symbols[pair.0 as usize].clone(),
        b: symbols[pair.1 as usize].clone(),
    };
    let mut heap: BinaryHeap<Entry> = pair_counts
        .iter()
        .map(|(&p, &c)| entry(&symbols, p, c))
        .collect();

    let mut merges = Vec::with_capacity(num_merges);
    for _ in 0..num_merges {
        // Lazy deletion: an entry is valid only if its count is current.
        let mut best = None;
        while let Some(e) = heap.pop() {
            if pair_counts.get(&e.pair) == Some(&e.count) {
                best = Some(e);
                break;
            }
        }
        let Some(best) = best else { break };

        let new_id = symbols.len() as u32;
        symbols.push(Arc::from(format!("{}{}", best.a, best.b)));

        let mut changed: FxHashSet<Pair> = FxHashSet::default();
        for idx in pair_to_words.remove(&best.pair).unwrap_or_default() {
            let freq = freqs[idx as usize];
            let merged = merge_word(&seqs[idx as usize], best.pair, new_id, |p, d| {
                *pair_counts.entry(p).or_insert(0) += d * freq;
                if d > 0 {
                    pair_to_words.entry(p).or_default().insert(idx);
                }
                changed.insert(p);
            });
            if let Some(new_seq) = merged {
                seqs[idx as usize] = new_seq;
            }
        }

        for p in changed {
            let count = pair_counts[&p];
            if count > 0 {
                heap.push(entry(&symbols, p, count));
            } else {
                pair_counts.remove(&p);
            }
        }

        merges.push((best.a.to_string(), best.b.to_string(), best.count));
    }
    merges
}

/// Applies learned merges to pretokens, with a cache of already-encoded words.
#[pyclass]
struct Encoder {
    /// (left, right) -> (rank, merged id)
    ranks: FxHashMap<Pair, (u32, u32)>,
    cache: FxHashMap<Box<[u8]>, Box<[u32]>>,
    cache_limit: usize,
}

impl Encoder {
    /// Same procedure as the Python `_bpe_word`: repeatedly merge (all
    /// occurrences of) the lowest-ranked pair present in the word.
    fn bpe(&self, bytes: &[u8]) -> Vec<u32> {
        let mut word: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
        while word.len() > 1 {
            let best = word
                .windows(2)
                .filter_map(|w| self.ranks.get(&(w[0], w[1])).map(|&(r, id)| (r, (w[0], w[1]), id)))
                .min_by_key(|&(r, _, _)| r);
            let Some((_, pair, id)) = best else { break };
            word = merge_word(&word, pair, id, |_, _| {}).expect("pair is in the word");
        }
        word
    }

    fn encode_into(&mut self, token: &str, out: &mut Vec<u32>) {
        let bytes = token.as_bytes();
        if let Some(ids) = self.cache.get(bytes) {
            out.extend_from_slice(ids);
            return;
        }
        let ids = self.bpe(bytes);
        out.extend_from_slice(&ids);
        if self.cache.len() >= self.cache_limit {
            self.cache.clear();
        }
        self.cache.insert(bytes.into(), ids.into_boxed_slice());
    }
}

#[pymethods]
impl Encoder {
    /// `merges[i]` = (left id, right id, merged id) of the rank-i merge.
    #[new]
    #[pyo3(signature = (merges, cache_limit = 1 << 20))]
    fn new(merges: Vec<(u32, u32, u32)>, cache_limit: usize) -> Self {
        let ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, (a, b, id))| ((a, b), (rank as u32, id)))
            .collect();
        Encoder {
            ranks,
            cache: FxHashMap::default(),
            cache_limit,
        }
    }

    /// Encode a sequence of pretokens into one flat list of ids.
    fn encode_words(&mut self, words: Vec<String>) -> Vec<u32> {
        let mut out = Vec::with_capacity(words.len() * 2);
        for w in &words {
            self.encode_into(w, &mut out);
        }
        out
    }
}

#[pymodule]
fn webbot_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_class::<Encoder>()?;
    Ok(())
}
