"""precompute_aligned.py — cross-tokenizer activation precompute for N models.

Drives the pipeline at the text level:
  1. Stream text from FineWeb-Edu.
  2. Chunk the text stream into primary-tokenizer sequences of fixed length,
     decode each chunk back to text.
  3. For each chunk: tokenize with every model's own tokenizer.
  4. Forward each model (batched, right-padded; causal attention keeps
     real-position activations clean without needing an attention mask).
  5. Align the N token streams per chunk using
     src.tokenizer_align.align_n_models.
  6. Write aligned activations and per-model last-token IDs to shard memmaps.

Store format (per shard dir):
  last_tok_{slug}.npy   [N]
  model_{slug}.npy      [N, D_model]
  meta.json

store_config.json carries `alignment_mode = "greedy"` so the dataset loader
knows which format to read.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.precompute_activations import (
    ActivationExtractor,
    ModelSpec,
    SignalHandler,
    append_jsonl,
    atomic_write_json,
    np_dtype_from_string,
    read_manifest,
    setup_logging,
    slugify,
)
from src.tokenizer_align import align_n_models


@dataclass
class AlignedPreprocessConfig:
    model_names: List[str]
    hook_points: List[str]
    num_aligned_tokens: int
    primary_sequence_length: int
    batch_size: int
    shard_size_tokens: int
    output_dir: str
    dtype: str = "float16"
    token_dtype: str = "int32"
    device: str = "cuda"
    hf_dataset_name: str = "HuggingFaceFW/fineweb-edu"
    hf_dataset_config: str = "sample-100BT"
    hf_split: str = "train"
    streaming: bool = True
    trust_remote_code: bool = False
    prepend_bos: bool = False
    max_window: int = 16
    seed: int = 42


class AlignedShardWriter:
    def __init__(self, cfg: AlignedPreprocessConfig, model_specs: List[ModelSpec]):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir)
        self.shards_dir = self.output_dir / "shards"
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.store_config_path = self.output_dir / "store_config.json"
        self.model_specs = model_specs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        if not self.store_config_path.exists():
            atomic_write_json(self.store_config_path, {
                "alignment_mode": "greedy",
                "dtype": cfg.dtype,
                "token_dtype": cfg.token_dtype,
                "primary_sequence_length": cfg.primary_sequence_length,
                "batch_size": cfg.batch_size,
                "shard_size_tokens": cfg.shard_size_tokens,
                "max_window": cfg.max_window,
                "prepend_bos": cfg.prepend_bos,
                "models": [
                    {**asdict(spec), "hook_point": hp}
                    for spec, hp in zip(model_specs, cfg.hook_points)
                ],
                "dataset": {
                    "name": cfg.hf_dataset_name,
                    "config": cfg.hf_dataset_config,
                    "split": cfg.hf_split,
                    "streaming": cfg.streaming,
                },
            })

    def existing_shards(self):
        return read_manifest(self.manifest_path)

    def committed_token_count(self) -> int:
        rows = self.existing_shards()
        return int(rows[-1]["token_end"]) if rows else 0

    def next_shard_index(self) -> int:
        rows = self.existing_shards()
        return int(rows[-1]["shard_index"]) + 1 if rows else 0

    def shard_dir(self, shard_index: int) -> Path:
        return self.shards_dir / f"shard_{shard_index:06d}"

    def prepare_memmaps(self, shard_index: int, num_tokens: int):
        shard_dir = self.shard_dir(shard_index)
        shard_dir.mkdir(parents=True, exist_ok=True)
        last_tok_mms: Dict[str, np.memmap] = {}
        act_mms: Dict[str, np.memmap] = {}
        for spec in self.model_specs:
            last_tok_mms[spec.name] = np.lib.format.open_memmap(
                shard_dir / f"last_tok_{spec.slug}.npy",
                mode="w+",
                dtype=np_dtype_from_string(self.cfg.token_dtype),
                shape=(num_tokens,),
            )
            act_mms[spec.name] = np.lib.format.open_memmap(
                shard_dir / f"model_{spec.slug}.npy",
                mode="w+",
                dtype=np_dtype_from_string(self.cfg.dtype),
                shape=(num_tokens, spec.hidden_size),
            )
        return shard_dir, last_tok_mms, act_mms

    def commit_shard(self, meta: dict, shard_dir: Path):
        atomic_write_json(shard_dir / "meta.json", meta)
        append_jsonl(self.manifest_path, meta)
        atomic_write_json(self.state_path, {
            "last_committed_shard_index": meta["shard_index"],
            "committed_tokens": meta["token_end"],
            "updated_at_unix": time.time(),
        })


class TextChunker:
    """Yields text chunks by packing whole paragraphs.

    For each document, split on blank-line paragraph boundaries, then greedily
    pack paragraphs (joined by "\\n\\n") into chunks of at most
    `primary_sequence_length` primary-tokenizer tokens. A paragraph that does
    not fit in the current chunk starts the next chunk; chunks never span
    document boundaries. Paragraphs longer than seq_len on their own are
    dropped (rare in practice).
    """

    _PARA_RE = re.compile(r"\n+")

    def __init__(self, cfg: AlignedPreprocessConfig, primary_tokenizer):
        self.cfg = cfg
        self.tok = primary_tokenizer
        self.seq_len = cfg.primary_sequence_length
        self._para_sep_n = len(self._enc("\n"))

    def _iter_text(self) -> Iterator[str]:
        ds = load_dataset(
            self.cfg.hf_dataset_name,
            self.cfg.hf_dataset_config,
            split=self.cfg.hf_split,
            streaming=self.cfg.streaming,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        for row in ds:
            text = row.get("text", None)
            if text:
                yield text

    def _enc(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def iter_chunks(self) -> Iterator[str]:
        for doc in self._iter_text():
            paragraphs = [p.strip() for p in self._PARA_RE.split(doc) if p.strip()]
            cur_text = ""
            cur_len = 0
            for para in paragraphs:
                n = len(self._enc(para))
                if n > self.seq_len:
                    continue
                sep_n = self._para_sep_n if cur_text else 0
                if cur_len + sep_n + n > self.seq_len:
                    yield cur_text
                    cur_text, cur_len = para, n
                else:
                    cur_text = (cur_text + "\n" + para) if cur_text else para
                    cur_len += sep_n + n
            if cur_text:
                yield cur_text


class AlignedActivationPreprocessor:
    def __init__(self, cfg: AlignedPreprocessConfig):
        self.cfg = cfg
        self.signal_handler = SignalHandler()
        self.extractors: Dict[str, ActivationExtractor] = {}
        self.model_specs: List[ModelSpec] = []
        self.tokenizers: List = []
        self.writer: Optional[AlignedShardWriter] = None

    def _load_models(self):
        for name, hp in zip(self.cfg.model_names, self.cfg.hook_points):
            self.extractors[name] = ActivationExtractor(
                model_name=name,
                hook_point=hp,
                device=self.cfg.device,
                precision=self.cfg.dtype,
            )
        self.tokenizers = [self.extractors[n].model.tokenizer for n in self.cfg.model_names]
        self.model_specs = [
            ModelSpec(name=name, slug=slugify(name), hidden_size=self.extractors[name].hidden_size)
            for name in self.cfg.model_names
        ]
        self.writer = AlignedShardWriter(self.cfg, self.model_specs)

    def _forward_batch(self, model_name: str, token_lists: List[List[int]]) -> np.ndarray:
        """Right-pad and forward once. Returns [B, S_max, D] float32 numpy (real positions only are valid)."""
        extractor = self.extractors[model_name]
        tokenizer = extractor.model.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0
        max_len = max((len(t) for t in token_lists), default=0)
        if max_len == 0:
            return np.empty((0, 0, extractor.hidden_size), dtype=np.float32)
        padded = np.full((len(token_lists), max_len), pad_id, dtype=np.int64)
        for i, t in enumerate(token_lists):
            padded[i, : len(t)] = t
        batch_tokens = torch.from_numpy(padded).to(self.cfg.device)
        acts = extractor.extract_batch(batch_tokens)  # torch on CPU, storage dtype
        return acts.float().numpy()  # fp32 for alignment/indexing safety

    def run(self):
        self._load_models()
        assert self.writer is not None

        committed = self.writer.committed_token_count()
        if committed >= self.cfg.num_aligned_tokens:
            logging.info("Already complete: %d committed >= %d target.", committed, self.cfg.num_aligned_tokens)
            return

        chunker = TextChunker(self.cfg, self.tokenizers[0]).iter_chunks()

        target_total = self.cfg.num_aligned_tokens
        remaining = target_total - committed
        pbar = tqdm(total=remaining, desc="aligned_tokens", unit="tok")
        shard_index = self.writer.next_shard_index()
        global_cursor = committed

        storage_dtype = np_dtype_from_string(self.cfg.dtype)
        token_dtype = np_dtype_from_string(self.cfg.token_dtype)
        failed_alignments = 0
        total_chunks = 0

        while pbar.n < remaining:
            if self.signal_handler.stop:
                logging.warning("Stop requested; exiting before new shard.")
                break

            shard_target = min(self.cfg.shard_size_tokens, remaining - pbar.n)
            shard_dir, last_tok_mms, act_mms = self.writer.prepare_memmaps(shard_index, shard_target)
            shard_cursor = 0
            shard_start = global_cursor
            stream_exhausted = False

            while shard_cursor < shard_target:
                chunk_texts: List[str] = []
                for _ in range(self.cfg.batch_size):
                    try:
                        chunk_texts.append(next(chunker))
                    except StopIteration:
                        stream_exhausted = True
                        break
                if not chunk_texts:
                    break

                # Per-model tokenization for this batch.
                per_model_tokens: List[List[List[int]]] = []
                for tk in self.tokenizers:
                    tok_lists = []
                    for text in chunk_texts:
                        ids = tk.encode(text, add_special_tokens=False)
                        if self.cfg.prepend_bos and getattr(tk, "bos_token_id", None) is not None:
                            ids = [tk.bos_token_id] + ids
                        tok_lists.append(ids)
                    per_model_tokens.append(tok_lists)

                # Per-model batched forward.
                per_model_acts: List[np.ndarray] = [
                    self._forward_batch(name, per_model_tokens[m_idx])
                    for m_idx, name in enumerate(self.cfg.model_names)
                ]

                # Per-chunk N-way alignment.
                for b_idx in range(len(chunk_texts)):
                    total_chunks += 1
                    streams = []
                    skip = False
                    for m_idx in range(len(self.cfg.model_names)):
                        toks = per_model_tokens[m_idx][b_idx]
                        if not toks:
                            skip = True
                            break
                        acts = per_model_acts[m_idx][b_idx, : len(toks), :]
                        streams.append((toks, acts, self.tokenizers[m_idx]))
                    if skip:
                        continue

                    res = align_n_models(streams, max_window=self.cfg.max_window)
                    if not res.completed:
                        failed_alignments += 1
                    n_aligned = res.aligned[0].shape[0]
                    if n_aligned == 0:
                        continue

                    free = shard_target - shard_cursor
                    if free <= 0:
                        break
                    to_write = min(n_aligned, free)

                    for m_idx, spec in enumerate(self.model_specs):
                        act_mms[spec.name][shard_cursor : shard_cursor + to_write, :] = (
                            res.aligned[m_idx][:to_write, :].astype(storage_dtype, copy=False)
                        )
                        last_tok_mms[spec.name][shard_cursor : shard_cursor + to_write] = (
                            res.last_tok[m_idx][:to_write].astype(token_dtype, copy=False)
                        )
                    shard_cursor += to_write
                    global_cursor += to_write
                    pbar.update(to_write)
                    if shard_cursor >= shard_target:
                        break

                if stream_exhausted:
                    break

            for mm in act_mms.values():
                mm.flush()
            for mm in last_tok_mms.values():
                mm.flush()

            if shard_cursor == 0:
                try:
                    for fp in shard_dir.iterdir():
                        fp.unlink()
                    shard_dir.rmdir()
                except Exception:
                    logging.exception("Failed to clean empty shard directory: %s", shard_dir)
                break

            actual = shard_cursor
            meta = {
                "shard_index": shard_index,
                "token_start": shard_start,
                "token_end": shard_start + actual,
                "num_tokens": actual,
                "alignment_mode": "greedy",
                "dtype": self.cfg.dtype,
                "token_dtype": self.cfg.token_dtype,
                "models": [
                    {"name": s.name, "slug": s.slug, "hidden_size": s.hidden_size, "hook_point": hp}
                    for s, hp in zip(self.model_specs, self.cfg.hook_points)
                ],
            }
            self.writer.commit_shard(meta, shard_dir)
            logging.info("Committed shard=%d aligned_range=[%d, %d)", shard_index, shard_start, shard_start + actual)
            shard_index += 1

            if stream_exhausted and shard_cursor < shard_target:
                logging.warning("Text stream exhausted; final shard is short.")
                break

        pbar.close()
        rate = 1.0 - (failed_alignments / total_chunks) if total_chunks else 0.0
        logging.info(
            "Done. committed=%d | chunks=%d | partial_align=%d (%.2f%% fully completed)",
            global_cursor, total_chunks, failed_alignments, rate * 100.0,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute aligned N-model activations for cross-tokenizer universal SAE.")
    p.add_argument("--model-names", nargs="+", required=True)
    p.add_argument("--hook-points", nargs="+", required=True,
                   help="One hook point per model (same order as --model-names), OR exactly one to apply to all.")
    p.add_argument("--num-aligned-tokens", type=int, required=True)
    p.add_argument("--primary-sequence-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size-tokens", type=int, default=262144)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--token-dtype", type=str, default="int32", choices=["int32", "int64"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf-dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--hf-dataset-config", type=str, default="sample-100BT")
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prepend-bos", action="store_true")
    p.add_argument("--max-window", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    setup_logging()
    args = parse_args()
    if len(args.hook_points) == 1 and len(args.model_names) > 1:
        args.hook_points = args.hook_points * len(args.model_names)
    if len(args.hook_points) != len(args.model_names):
        raise ValueError(
            "--hook-points must have the same length as --model-names (or exactly one).")

    cfg = AlignedPreprocessConfig(
        model_names=args.model_names,
        hook_points=args.hook_points,
        num_aligned_tokens=args.num_aligned_tokens,
        primary_sequence_length=args.primary_sequence_length,
        batch_size=args.batch_size,
        shard_size_tokens=args.shard_size_tokens,
        output_dir=args.output_dir,
        dtype=args.dtype,
        token_dtype=args.token_dtype,
        device=args.device,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        streaming=not args.no_streaming,
        trust_remote_code=args.trust_remote_code,
        prepend_bos=args.prepend_bos,
        max_window=args.max_window,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    AlignedActivationPreprocessor(cfg).run()


if __name__ == "__main__":
    main()
