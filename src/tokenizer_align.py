"""Cross-Model Activation Alignment (Algorithm 1 of Jiralerspong & Bricken, arXiv:2602.11729).

Given per-model token sequences and per-token hidden activations, match
positions between models with different tokenizers by decoding text windows.
On a decoded-text match, keep the activation at the final token of each
window (many-to-one compression relying on self-attention to have summarised
the span).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


_WS_RE = re.compile(r"\s+")


def default_normalize(s: str) -> str:
    """NFKC → casefold → collapse internal whitespace → strip."""
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s)
    return s.casefold().strip()


def default_is_non_content(tokenizer) -> Callable[[int], bool]:
    """Token is non-content if it's a special id or decodes to pure whitespace."""
    special = set(int(t) for t in getattr(tokenizer, "all_special_ids", []) or [])

    def check(tok_id: int) -> bool:
        if int(tok_id) in special:
            return True
        decoded = tokenizer.decode([int(tok_id)])
        return decoded == "" or decoded.strip() == ""

    return check


@dataclass
class AlignResult:
    aligned_a: np.ndarray      # [N, D_a]
    aligned_b: np.ndarray      # [N, D_b]
    last_tok_a: np.ndarray     # [N] int64 — final token id of each matched window
    last_tok_b: np.ndarray     # [N] int64
    completed: bool            # False iff aligner returned early on irreconcilable divergence


def align_two_models(
    tokens_a: Sequence[int],
    activations_a: np.ndarray,
    tokenizer_a,
    tokens_b: Sequence[int],
    activations_b: np.ndarray,
    tokenizer_b,
    *,
    is_non_content_a: Optional[Callable[[int], bool]] = None,
    is_non_content_b: Optional[Callable[[int], bool]] = None,
    normalize: Optional[Callable[[str], str]] = None,
    max_window: int = 16,
) -> AlignResult:
    tokens_a = [int(t) for t in tokens_a]
    tokens_b = [int(t) for t in tokens_b]
    if activations_a.shape[0] != len(tokens_a):
        raise ValueError(f"activations_a rows ({activations_a.shape[0]}) != tokens_a len ({len(tokens_a)})")
    if activations_b.shape[0] != len(tokens_b):
        raise ValueError(f"activations_b rows ({activations_b.shape[0]}) != tokens_b len ({len(tokens_b)})")

    if is_non_content_a is None:
        is_non_content_a = default_is_non_content(tokenizer_a)
    if is_non_content_b is None:
        is_non_content_b = default_is_non_content(tokenizer_b)
    if normalize is None:
        normalize = default_normalize

    def decode_a(lo: int, hi: int) -> str:
        return tokenizer_a.decode(tokens_a[lo:hi])

    def decode_b(lo: int, hi: int) -> str:
        return tokenizer_b.decode(tokens_b[lo:hi])

    ta, tb = len(tokens_a), len(tokens_b)
    out_idx_a: list[int] = []
    out_idx_b: list[int] = []
    p_a, p_b = 0, 0
    completed = True

    while p_a < ta and p_b < tb:
        while p_a < ta and is_non_content_a(tokens_a[p_a]):
            p_a += 1
        while p_b < tb and is_non_content_b(tokens_b[p_b]):
            p_b += 1
        if p_a >= ta or p_b >= tb:
            break

        s_a = normalize(decode_a(p_a, p_a + 1))
        s_b = normalize(decode_b(p_b, p_b + 1))
        if s_a == s_b:
            out_idx_a.append(p_a)
            out_idx_b.append(p_b)
            p_a += 1
            p_b += 1
            continue

        e_a, e_b = p_a + 1, p_b + 1
        found = False
        while True:
            w_a = normalize(decode_a(p_a, e_a))
            w_b = normalize(decode_b(p_b, e_b))
            if w_a == w_b:
                out_idx_a.append(e_a - 1)
                out_idx_b.append(e_b - 1)
                p_a, p_b = e_a, e_b
                found = True
                break
            if (e_a - p_a) > max_window or (e_b - p_b) > max_window:
                break
            a_can = e_a < ta
            b_can = e_b < tb
            if len(w_a) < len(w_b) and a_can:
                e_a += 1
            elif b_can:
                e_b += 1
            elif a_can:
                e_a += 1
            else:
                break

        if not found:
            completed = False
            break

    d_a = activations_a.shape[1] if activations_a.ndim == 2 else 0
    d_b = activations_b.shape[1] if activations_b.ndim == 2 else 0
    if out_idx_a:
        aligned_a = activations_a[np.asarray(out_idx_a, dtype=np.int64)].copy()
        aligned_b = activations_b[np.asarray(out_idx_b, dtype=np.int64)].copy()
        last_tok_a = np.asarray([tokens_a[i] for i in out_idx_a], dtype=np.int64)
        last_tok_b = np.asarray([tokens_b[i] for i in out_idx_b], dtype=np.int64)
    else:
        aligned_a = np.empty((0, d_a), dtype=activations_a.dtype)
        aligned_b = np.empty((0, d_b), dtype=activations_b.dtype)
        last_tok_a = np.empty((0,), dtype=np.int64)
        last_tok_b = np.empty((0,), dtype=np.int64)

    return AlignResult(
        aligned_a=aligned_a,
        aligned_b=aligned_b,
        last_tok_a=last_tok_a,
        last_tok_b=last_tok_b,
        completed=completed,
    )


@dataclass
class MultiAlignResult:
    aligned: list         # list of np.ndarray, one per stream, each [N, D_i]
    last_tok: list        # list of np.ndarray, one per stream, each [N] int64
    completed: bool


def align_n_models(
    streams,                # list of (tokens, activations, tokenizer) triples
    *,
    is_non_content_fns: Optional[list] = None,
    normalize: Optional[Callable[[str], str]] = None,
    max_window: int = 16,
) -> MultiAlignResult:
    """N-stream extension of Algorithm 1. At each step, skip non-content tokens,
    try a 1-to-1 decoded-text match across all N streams; on mismatch, grow the
    stream with the shortest decoded text. Match when all N decoded texts are
    normalize-equal.
    """
    n = len(streams)
    if n < 2:
        raise ValueError("need at least 2 streams")

    tokens_list = [[int(t) for t in s[0]] for s in streams]
    acts_list = [s[1] for s in streams]
    tokenizers = [s[2] for s in streams]
    for i, (toks, acts) in enumerate(zip(tokens_list, acts_list)):
        if acts.shape[0] != len(toks):
            raise ValueError(f"stream {i}: activations rows ({acts.shape[0]}) != tokens len ({len(toks)})")

    if is_non_content_fns is None:
        is_non_content_fns = [default_is_non_content(tk) for tk in tokenizers]
    if normalize is None:
        normalize = default_normalize

    lens = [len(t) for t in tokens_list]

    def decode(i: int, lo: int, hi: int) -> str:
        return tokenizers[i].decode(tokens_list[i][lo:hi])

    p = [0] * n
    out_idx: list = [[] for _ in range(n)]
    completed = True

    while all(p[i] < lens[i] for i in range(n)):
        for i in range(n):
            while p[i] < lens[i] and is_non_content_fns[i](tokens_list[i][p[i]]):
                p[i] += 1
        if any(p[i] >= lens[i] for i in range(n)):
            break

        singles = [normalize(decode(i, p[i], p[i] + 1)) for i in range(n)]
        if all(s == singles[0] for s in singles[1:]):
            for i in range(n):
                out_idx[i].append(p[i])
                p[i] += 1
            continue

        e = [p[i] + 1 for i in range(n)]
        found = False
        while True:
            ws = [normalize(decode(i, p[i], e[i])) for i in range(n)]
            if all(w == ws[0] for w in ws[1:]):
                for i in range(n):
                    out_idx[i].append(e[i] - 1)
                    p[i] = e[i]
                found = True
                break
            if any((e[i] - p[i]) > max_window for i in range(n)):
                break
            # Grow the stream with the shortest decoded text; if it cannot
            # expand, pick the next shortest that can; if none can, break.
            order = sorted(range(n), key=lambda i: len(ws[i]))
            expanded = False
            for i in order:
                if e[i] < lens[i]:
                    e[i] += 1
                    expanded = True
                    break
            if not expanded:
                break

        if not found:
            completed = False
            break

    aligned = []
    last_tok = []
    for i in range(n):
        d = acts_list[i].shape[1] if acts_list[i].ndim == 2 else 0
        if out_idx[i]:
            idx = np.asarray(out_idx[i], dtype=np.int64)
            aligned.append(acts_list[i][idx].copy())
            last_tok.append(np.asarray([tokens_list[i][j] for j in out_idx[i]], dtype=np.int64))
        else:
            aligned.append(np.empty((0, d), dtype=acts_list[i].dtype))
            last_tok.append(np.empty((0,), dtype=np.int64))

    return MultiAlignResult(aligned=aligned, last_tok=last_tok, completed=completed)


if __name__ == "__main__":
    class MockTokenizer:
        def __init__(self, vocab: dict[int, str], special_ids: Sequence[int] = ()):
            self.vocab = vocab
            self.all_special_ids = list(special_ids)

        def decode(self, ids):
            if isinstance(ids, (int, np.integer)):
                ids = [int(ids)]
            return "".join(self.vocab.get(int(i), "") for i in ids)

    def _acts(n, d, seed):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, d)).astype(np.float32)

    # Test 1: identical tokenizers, no special tokens
    tok = MockTokenizer({0: "a", 1: "b", 2: "c"})
    ha, hb = _acts(3, 4, 1), _acts(3, 4, 2)
    r = align_two_models([0, 1, 2], ha, tok, [0, 1, 2], hb, tok)
    assert r.aligned_a.shape == (3, 4) and r.aligned_b.shape == (3, 4)
    assert np.array_equal(r.aligned_a, ha) and np.array_equal(r.aligned_b, hb)
    assert list(r.last_tok_a) == [0, 1, 2] and list(r.last_tok_b) == [0, 1, 2]
    assert r.completed
    print("test 1 (identical tokenizers) ok")

    # Test 2: canonical 1989 case
    tok_a = MockTokenizer({100: "1989"})
    tok_b = MockTokenizer({200: "198", 201: "9"})
    ha, hb = _acts(1, 3, 3), _acts(2, 3, 4)
    r = align_two_models([100], ha, tok_a, [200, 201], hb, tok_b)
    assert r.aligned_a.shape == (1, 3) and r.aligned_b.shape == (1, 3)
    assert np.array_equal(r.aligned_a[0], ha[0])
    assert np.array_equal(r.aligned_b[0], hb[1])          # last of window on B side
    assert list(r.last_tok_a) == [100] and list(r.last_tok_b) == [201]
    assert r.completed
    print("test 2 (1989 case) ok")

    # Test 3: both sides split differently
    tok_a = MockTokenizer({10: "hello", 11: " world"})
    tok_b = MockTokenizer({20: "hel", 21: "lo", 22: " world"})
    ha, hb = _acts(2, 2, 5), _acts(3, 2, 6)
    r = align_two_models([10, 11], ha, tok_a, [20, 21, 22], hb, tok_b)
    assert r.aligned_a.shape == (2, 2) and r.aligned_b.shape == (2, 2)
    assert np.array_equal(r.aligned_a[0], ha[0]) and np.array_equal(r.aligned_b[0], hb[1])
    assert np.array_equal(r.aligned_a[1], ha[1]) and np.array_equal(r.aligned_b[1], hb[2])
    assert list(r.last_tok_a) == [10, 11] and list(r.last_tok_b) == [21, 22]
    assert r.completed
    print("test 3 (both-sides split) ok")

    # Test 4: non-content skip (special id on A side)
    tok_a = MockTokenizer({999: "<bos>", 10: "hello"}, special_ids=[999])
    tok_b = MockTokenizer({10: "hello"})
    ha, hb = _acts(2, 2, 7), _acts(1, 2, 8)
    r = align_two_models([999, 10], ha, tok_a, [10], hb, tok_b)
    assert r.aligned_a.shape == (1, 2)
    assert np.array_equal(r.aligned_a[0], ha[1]) and np.array_equal(r.aligned_b[0], hb[0])
    assert list(r.last_tok_a) == [10] and list(r.last_tok_b) == [10]
    assert r.completed
    print("test 4 (non-content skip) ok")

    # Test 5: irreconcilable divergence
    tok_a = MockTokenizer({10: "hello"})
    tok_b = MockTokenizer({20: "xyz"})
    ha, hb = _acts(1, 2, 9), _acts(1, 2, 10)
    r = align_two_models([10], ha, tok_a, [20], hb, tok_b)
    assert r.aligned_a.shape == (0, 2) and r.aligned_b.shape == (0, 2)
    assert not r.completed
    print("test 5 (irreconcilable) ok")

    # Test 5b: partial alignment before divergence
    tok_a = MockTokenizer({10: "hello", 30: "QQ"})
    tok_b = MockTokenizer({10: "hello", 40: "PP"})
    ha, hb = _acts(2, 2, 11), _acts(2, 2, 12)
    r = align_two_models([10, 30], ha, tok_a, [10, 40], hb, tok_b)
    assert r.aligned_a.shape == (1, 2) and not r.completed
    assert np.array_equal(r.aligned_a[0], ha[0]) and np.array_equal(r.aligned_b[0], hb[0])
    print("test 5b (partial-then-diverge) ok")

    # Test 6: max_window cap
    tok_a = MockTokenizer({i: "a" for i in range(20)})
    tok_b = MockTokenizer({100: "a" * 20})
    # A: 20 'a' tokens, B: 1 token decoding to "aaaa...a"; needs window of 20 on A — capped at 16
    r = align_two_models(list(range(20)), _acts(20, 1, 13), tok_a, [100], _acts(1, 1, 14), tok_b, max_window=16)
    assert r.aligned_a.shape == (0, 1) and not r.completed
    print("test 6 (max_window cap) ok")

    # Normalization sanity
    assert default_normalize("HELLO") == "hello"                 # casefold
    assert default_normalize("  hello   world  ") == "hello world"  # ws collapse + strip
    assert default_normalize("ﬁle") == "file"                    # NFKC: ligature → 'fi'
    print("normalize (casefold / ws / NFKC) ok")

    # Test 7: 3-model identical tokenizers
    tok = MockTokenizer({0: "a", 1: "b", 2: "c"})
    ha, hb, hc = _acts(3, 4, 15), _acts(3, 4, 16), _acts(3, 4, 17)
    r = align_n_models([([0, 1, 2], ha, tok), ([0, 1, 2], hb, tok), ([0, 1, 2], hc, tok)])
    assert r.completed and [a.shape for a in r.aligned] == [(3, 4), (3, 4), (3, 4)]
    assert np.array_equal(r.aligned[0], ha)
    print("test 7 (3 models identical) ok")

    # Test 8: three different tokenizations of "1989"
    # A: ["1989"]; B: ["198", "9"]; C: ["1", "989"]
    tok_a = MockTokenizer({100: "1989"})
    tok_b = MockTokenizer({200: "198", 201: "9"})
    tok_c = MockTokenizer({300: "1", 301: "989"})
    ha, hb, hc = _acts(1, 2, 18), _acts(2, 2, 19), _acts(2, 2, 20)
    r = align_n_models([([100], ha, tok_a), ([200, 201], hb, tok_b), ([300, 301], hc, tok_c)])
    assert r.completed and [a.shape for a in r.aligned] == [(1, 2), (1, 2), (1, 2)]
    assert np.array_equal(r.aligned[0][0], ha[0])
    assert np.array_equal(r.aligned[1][0], hb[1])   # last of window
    assert np.array_equal(r.aligned[2][0], hc[1])   # last of window
    assert list(r.last_tok[0]) == [100] and list(r.last_tok[1]) == [201] and list(r.last_tok[2]) == [301]
    print("test 8 (3-way 1989 case) ok")

    # Test 9: 3-way irreconcilable — A,B match but C diverges
    tok_a = MockTokenizer({10: "hello"})
    tok_b = MockTokenizer({20: "hello"})
    tok_c = MockTokenizer({30: "xyz"})
    r = align_n_models([
        ([10], _acts(1, 2, 21), tok_a),
        ([20], _acts(1, 2, 22), tok_b),
        ([30], _acts(1, 2, 23), tok_c),
    ])
    assert not r.completed and [a.shape for a in r.aligned] == [(0, 2), (0, 2), (0, 2)]
    print("test 9 (3-way irreconcilable) ok")

    print("\nall tests passed")
