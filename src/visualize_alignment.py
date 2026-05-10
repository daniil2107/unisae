"""Visualize cross-tokenizer alignment on one real FineWeb-Edu document.

Streams a single document from FineWeb-Edu, applies our paragraph-pack chunker,
tokenizes the first chunk with three tokenizers, and walks the alignment
algorithm inline so we can capture each matched window's decoded text per model.

Run:
    conda run -n ttsae python -m src.visualize_alignment
"""

from __future__ import annotations

import re
from typing import Iterator, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from src.tokenizer_align import default_is_non_content, default_normalize


_PARA_RE = re.compile(r"\n+")


def _iter_chunks(text: str, primary_tokenizer, seq_len: int) -> Iterator[str]:
    """Mirror of src.precompute_aligned.TextChunker, inlined to avoid pulling in
    transformer_lens via that module's imports."""
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    para_sep_n = len(primary_tokenizer.encode("\n", add_special_tokens=False))
    cur_text = ""
    cur_len = 0
    for para in paragraphs:
        n = len(primary_tokenizer.encode(para, add_special_tokens=False))
        if n > seq_len:
            continue
        sep_n = para_sep_n if cur_text else 0
        if cur_len + sep_n + n > seq_len:
            yield cur_text
            cur_text, cur_len = para, n
        else:
            cur_text = (cur_text + "\n" + para) if cur_text else para
            cur_len += sep_n + n
    if cur_text:
        yield cur_text


MODEL_NAMES = ["gpt2", "EleutherAI/pythia-160m", "HuggingFaceTB/SmolLM-135M"]
SHORT_NAMES = ["gpt2", "pythia", "smollm"]
PRIMARY_SEQ_LEN = 512
RAW_TEXT_PREVIEW = 1500
MAX_WINDOW = 16


def _fetch_one_doc() -> str:
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-100BT",
        split="train",
        streaming=True,
    )
    for row in ds:
        text = row.get("text", "")
        if text and len(text) > 1000:
            return text
    raise RuntimeError("no suitable doc found")


def _first_chunk(text: str, primary_tokenizer) -> str:
    for chunk in _iter_chunks(text, primary_tokenizer, PRIMARY_SEQ_LEN):
        return chunk
    raise RuntimeError("chunker yielded nothing")


def _walk_alignment(token_lists, tokenizers) -> Tuple[List[List[str]], bool]:
    """Re-implements the align_n_models loop, but records the decoded text of
    each matched window per stream. Returns (windows_per_position, completed)
    where windows_per_position[k] = [text_for_model_0, text_for_model_1, ...]."""
    n = len(token_lists)
    is_nc = [default_is_non_content(t) for t in tokenizers]
    lens = [len(t) for t in token_lists]

    def decode(i: int, lo: int, hi: int) -> str:
        return tokenizers[i].decode(token_lists[i][lo:hi])

    p = [0] * n
    rows: List[List[str]] = []
    completed = True

    while all(p[i] < lens[i] for i in range(n)):
        for i in range(n):
            while p[i] < lens[i] and is_nc[i](token_lists[i][p[i]]):
                p[i] += 1
        if any(p[i] >= lens[i] for i in range(n)):
            break

        singles = [default_normalize(decode(i, p[i], p[i] + 1)) for i in range(n)]
        if all(s == singles[0] for s in singles[1:]):
            rows.append([decode(i, p[i], p[i] + 1) for i in range(n)])
            for i in range(n):
                p[i] += 1
            continue

        e = [p[i] + 1 for i in range(n)]
        found = False
        while True:
            ws = [default_normalize(decode(i, p[i], e[i])) for i in range(n)]
            if all(w == ws[0] for w in ws[1:]):
                rows.append([decode(i, p[i], e[i]) for i in range(n)])
                for i in range(n):
                    p[i] = e[i]
                found = True
                break
            if any((e[i] - p[i]) > MAX_WINDOW for i in range(n)):
                break
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

    return rows, completed


def _format_window(window_text: str, sub_tokens: List[str]) -> str:
    """Show '·'-joined sub-tokens when a window contains multiple tokens."""
    if len(sub_tokens) == 1:
        return repr(window_text)
    return "·".join(repr(t) for t in sub_tokens)


def main():
    print("Loading tokenizers...")
    tokenizers = [AutoTokenizer.from_pretrained(n) for n in MODEL_NAMES]

    print("Streaming one doc from FineWeb-Edu...")
    raw = _fetch_one_doc()

    print("\n" + "=" * 80)
    print(f"RAW TEXT (first {RAW_TEXT_PREVIEW} chars of {len(raw)} total)")
    print("=" * 80)
    print(raw[:RAW_TEXT_PREVIEW] + ("..." if len(raw) > RAW_TEXT_PREVIEW else ""))

    chunk = _first_chunk(raw, tokenizers[0])
    counts = [len(t.encode(chunk, add_special_tokens=False)) for t in tokenizers]

    print("\n" + "=" * 80)
    print(f"FIRST CHUNK (gpt2={counts[0]} tok, pythia={counts[1]} tok, smollm={counts[2]} tok)")
    print("=" * 80)
    print(chunk)

    token_lists = [t.encode(chunk, add_special_tokens=False) for t in tokenizers]
    rows, completed = _walk_alignment(token_lists, tokenizers)

    # Recompute per-window sub-tokens by re-running pointer walk (so we can show splits)
    # Easier: replay the algorithm and record token-id windows alongside text.
    sub_token_rows: List[List[List[str]]] = []
    is_nc = [default_is_non_content(t) for t in tokenizers]
    p = [0, 0, 0]
    for _ in rows:
        for i in range(3):
            while p[i] < len(token_lists[i]) and is_nc[i](token_lists[i][p[i]]):
                p[i] += 1
        singles = [default_normalize(tokenizers[i].decode([token_lists[i][p[i]]])) for i in range(3)]
        if all(s == singles[0] for s in singles[1:]):
            sub_token_rows.append([[tokenizers[i].decode([token_lists[i][p[i]]])] for i in range(3)])
            for i in range(3):
                p[i] += 1
            continue
        e = [p[i] + 1 for i in range(3)]
        while True:
            ws = [default_normalize(tokenizers[i].decode(token_lists[i][p[i]:e[i]])) for i in range(3)]
            if all(w == ws[0] for w in ws[1:]):
                sub_token_rows.append([
                    [tokenizers[i].decode([tid]) for tid in token_lists[i][p[i]:e[i]]]
                    for i in range(3)
                ])
                for i in range(3):
                    p[i] = e[i]
                break
            order = sorted(range(3), key=lambda i: len(ws[i]))
            for i in order:
                if e[i] < len(token_lists[i]):
                    e[i] += 1
                    break
            else:
                break

    print("\n" + "=" * 80)
    print(f"ALIGNMENT — {len(rows)} positions ({'completed' if completed else 'truncated on divergence'})")
    print("=" * 80)
    header = f"{'idx':>4}  {'gpt2':<32}  {'pythia':<32}  {'smollm':<32}"
    print(header)
    print("-" * len(header))
    for k, (texts, subs) in enumerate(zip(rows, sub_token_rows)):
        cells = [_format_window(texts[i], subs[i]) for i in range(3)]
        cells = [c if len(c) <= 32 else c[:29] + "..." for c in cells]
        print(f"{k:>4}  {cells[0]:<32}  {cells[1]:<32}  {cells[2]:<32}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for sn, c in zip(SHORT_NAMES, counts):
        print(f"  {sn:8s} pre-align tokens: {c}")
    print(f"  aligned positions:        {len(rows)}")
    print(f"  retention vs shortest:    {len(rows) / max(min(counts), 1):.2%}")
    print(f"  completed:                {completed}")


if __name__ == "__main__":
    main()
