#!/usr/bin/env python3
"""
Precompute aligned hidden activations from one or more TransformerLens models
over streamed FineWeb-Edu text and store them in shard-based mmap-friendly files.

Example:
    python preprocess_activations.py \
      --model-names gpt2-small EleutherAI/pythia-160m \
      --hook-point blocks.6.hook_resid_post \
      --num-tokens 2000000 \
      --sequence-length 256 \
      --batch-size 8 \
      --shard-size-tokens 262144 \
      --output-dir ./activation_store \
      --dtype float16 \
      --num-workers 4 \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer


# -----------------------------
# Configs
# -----------------------------

@dataclass
class PreprocessConfig:
    model_names: List[str]
    hook_point: str
    num_tokens: int
    sequence_length: int
    batch_size: int
    shard_size_tokens: int
    output_dir: str
    dtype: str = "float16"          # float16, bfloat16, float32
    token_dtype: str = "int32"
    num_workers: int = 4
    device: str = "cuda"
    hf_dataset_name: str = "HuggingFaceFW/fineweb-edu"
    hf_dataset_config: str = "sample-100BT"
    hf_split: str = "train"
    streaming: bool = True
    add_bos: bool = False
    append_eos_between_docs: bool = True
    trust_remote_code: bool = False
    log_interval_batches: int = 50
    seed: int = 42


@dataclass
class ModelSpec:
    name: str
    slug: str
    hidden_size: int


@dataclass
class ShardMeta:
    shard_index: int
    token_start: int
    token_end: int
    num_tokens: int
    hook_point: str
    dtype: str
    token_dtype: str
    sequence_length: int
    batch_size: int
    models: List[Dict[str, object]]


# -----------------------------
# Utilities
# -----------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def slugify(s: str) -> str:
    s = s.strip().replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", s)
    return s


def np_dtype_from_string(dtype: str) -> np.dtype:
    mapping = {
        "float16": np.float16,
        "bfloat16": np.float16,  # stored as float16 unless you choose a different backend format
        "float32": np.float32,
        "int32": np.int32,
        "int64": np.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def torch_dtype_from_string(dtype: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def read_manifest(manifest_path: Path) -> List[dict]:
    if not manifest_path.exists():
        return []
    rows = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -----------------------------
# Graceful shutdown
# -----------------------------

class StopRequested(Exception):
    pass


class SignalHandler:
    def __init__(self) -> None:
        self.stop = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame) -> None:
        logging.warning("Received signal %s. Will stop after current shard commit.", signum)
        self.stop = True


# -----------------------------
# Token streaming
# -----------------------------

class StreamingTokenBuffer:
    """
    Streams text from FineWeb-Edu, tokenizes with a single aligned tokenizer,
    and yields contiguous fixed-length token sequences.

    Resume strategy:
      - If resuming, we skip 'skip_tokens' token positions by consuming from the stream.
      - This avoids recomputing finished shards, though the initial skip is linear in skipped tokens.
      - For truly gigantic restart scenarios, see performance recommendations below.
    """

    def __init__(
        self,
        cfg: PreprocessConfig,
        tokenizer,
        skip_tokens: int = 0,
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.skip_tokens = skip_tokens
        self._buffer: List[int] = []

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

    def _encode_doc(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        if self.cfg.append_eos_between_docs:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                ids = ids + [int(eos_id)]
        return ids

    def iter_token_sequences(self) -> Iterator[np.ndarray]:
        seq_len = self.cfg.sequence_length
        skipped = 0

        for text in self._iter_text():
            token_ids = self._encode_doc(text)
            self._buffer.extend(token_ids)

            # Resume: skip already materialized tokens.
            if skipped < self.skip_tokens:
                remaining_skip = self.skip_tokens - skipped
                to_skip_now = min(len(self._buffer), remaining_skip)
                if to_skip_now > 0:
                    del self._buffer[:to_skip_now]
                    skipped += to_skip_now

            while len(self._buffer) >= seq_len:
                seq = self._buffer[:seq_len]
                del self._buffer[:seq_len]
                yield np.asarray(seq, dtype=np.int32)


# -----------------------------
# Model wrapper
# -----------------------------

class ActivationExtractor:
    """
    Extracts only one requested hook activation per forward pass.
    Avoids storing full ActivationCache, which is much more memory-heavy.
    """

    def __init__(self, model_name: str, hook_point: str, device: str, precision: str):
        self.model_name = model_name
        self.hook_point = hook_point
        self.device = device
        self.precision = precision
        self.model = self._load_model(model_name, device)
        self.hidden_size = int(self.model.cfg.d_model)

    @staticmethod
    def _load_model(model_name: str, device: str) -> HookedTransformer:
        from transformers import AutoTokenizer
        logging.info("Loading model: %s", model_name)
        # Some tokenizers (e.g., Qwen2/2.5) ship configs that trip newer
        # transformers' strict validation when HookedTransformer loads them
        # with default args ("add_bos_token = True but bos_token = None").
        # Pre-loading with add_bos_token=False sidesteps it. We prepend BOS
        # ourselves at the encode site, so this is safe for all models.
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=False)
        except Exception:
            tokenizer = None
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            tokenizer=tokenizer,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @torch.inference_mode()
    def extract_batch(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, S] on model device
        returns: [B, S, D] on CPU, cast to requested storage dtype
        """
        captured = {}

        def save_hook(value: torch.Tensor, hook):
            captured["act"] = value.detach()

        # The forward output is ignored; only the hook output is needed.
        _ = self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_point, save_hook)],
        )

        if "act" not in captured:
            raise RuntimeError(
                f"Hook point '{self.hook_point}' was not captured for model '{self.model_name}'."
            )

        act = captured["act"]
        if act.ndim != 3:
            raise ValueError(
                f"Expected hook activation shape [B, S, D], got {tuple(act.shape)} "
                f"for model '{self.model_name}' at hook '{self.hook_point}'."
            )

        store_dtype = torch_dtype_from_string(self.precision)
        act = act.to(dtype=store_dtype).cpu().contiguous()
        return act


# -----------------------------
# Shard writer
# -----------------------------

class ShardWriter:
    def __init__(self, cfg: PreprocessConfig, model_specs: List[ModelSpec]) -> None:
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
            atomic_write_json(
                self.store_config_path,
                {
                    "hook_point": cfg.hook_point,
                    "dtype": cfg.dtype,
                    "token_dtype": cfg.token_dtype,
                    "sequence_length": cfg.sequence_length,
                    "batch_size": cfg.batch_size,
                    "shard_size_tokens": cfg.shard_size_tokens,
                    "models": [asdict(m) for m in model_specs],
                    "dataset": {
                        "name": cfg.hf_dataset_name,
                        "config": cfg.hf_dataset_config,
                        "split": cfg.hf_split,
                        "streaming": cfg.streaming,
                    },
                },
            )

    def existing_shards(self) -> List[dict]:
        return read_manifest(self.manifest_path)

    def committed_token_count(self) -> int:
        rows = self.existing_shards()
        if not rows:
            return 0
        return int(rows[-1]["token_end"])

    def next_shard_index(self) -> int:
        rows = self.existing_shards()
        if not rows:
            return 0
        return int(rows[-1]["shard_index"]) + 1

    def shard_dir(self, shard_index: int) -> Path:
        return self.shards_dir / f"shard_{shard_index:06d}"

    def prepare_memmaps(self, shard_index: int, token_start: int, num_tokens: int):
        shard_dir = self.shard_dir(shard_index)
        shard_dir.mkdir(parents=True, exist_ok=True)

        token_path = shard_dir / "tokens.npy"
        token_mm = np.lib.format.open_memmap(
            token_path,
            mode="w+",
            dtype=np_dtype_from_string(self.cfg.token_dtype),
            shape=(num_tokens,),
        )

        act_mms = {}
        for spec in self.model_specs:
            act_path = shard_dir / f"model_{spec.slug}.npy"
            act_mms[spec.name] = np.lib.format.open_memmap(
                act_path,
                mode="w+",
                dtype=np_dtype_from_string(self.cfg.dtype),
                shape=(num_tokens, spec.hidden_size),
            )

        return shard_dir, token_mm, act_mms

    def commit_shard(self, meta: ShardMeta, shard_dir: Path) -> None:
        meta_path = shard_dir / "meta.json"
        atomic_write_json(meta_path, asdict(meta))
        append_jsonl(self.manifest_path, asdict(meta))
        atomic_write_json(
            self.state_path,
            {
                "last_committed_shard_index": meta.shard_index,
                "committed_tokens": meta.token_end,
                "updated_at_unix": time.time(),
            },
        )


# -----------------------------
# Main preprocessing runner
# -----------------------------

class ActivationPreprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.signal_handler = SignalHandler()

        if cfg.shard_size_tokens % cfg.sequence_length != 0:
            raise ValueError("shard_size_tokens must be divisible by sequence_length.")
        if cfg.num_tokens % cfg.sequence_length != 0:
            logging.warning(
                "num_tokens is not divisible by sequence_length; final partial sequence will be dropped."
            )

        self.writer: Optional[ShardWriter] = None
        self.extractors: Dict[str, ActivationExtractor] = {}
        self.primary_tokenizer = None
        self.model_specs: List[ModelSpec] = []

    def _load_models(self) -> None:
        for model_name in self.cfg.model_names:
            extractor = ActivationExtractor(
                model_name=model_name,
                hook_point=self.cfg.hook_point,
                device=self.cfg.device,
                precision=self.cfg.dtype,
            )
            self.extractors[model_name] = extractor

        # Use the first model's tokenizer for corpus tokenization.
        first_model = self.extractors[self.cfg.model_names[0]].model
        self.primary_tokenizer = first_model.tokenizer

        self.model_specs = [
            ModelSpec(
                name=name,
                slug=slugify(name),
                hidden_size=self.extractors[name].hidden_size,
            )
            for name in self.cfg.model_names
        ]
        self.writer = ShardWriter(self.cfg, self.model_specs)

    def run(self) -> None:
        self._load_models()
        assert self.writer is not None
        assert self.primary_tokenizer is not None

        committed_tokens = self.writer.committed_token_count()
        if committed_tokens >= self.cfg.num_tokens:
            logging.info(
                "Nothing to do. Existing store already contains %d tokens (target=%d).",
                committed_tokens,
                self.cfg.num_tokens,
            )
            return

        logging.info("Resuming from committed token count: %d", committed_tokens)

        token_stream = StreamingTokenBuffer(
            cfg=self.cfg,
            tokenizer=self.primary_tokenizer,
            skip_tokens=committed_tokens,
        ).iter_token_sequences()

        remaining = self.cfg.num_tokens - committed_tokens
        full_sequences_needed = remaining // self.cfg.sequence_length
        total_effective_tokens = full_sequences_needed * self.cfg.sequence_length

        pbar = tqdm(total=total_effective_tokens, desc="preprocess_tokens", unit="tok")
        shard_index = self.writer.next_shard_index()
        global_token_cursor = committed_tokens

        while pbar.n < total_effective_tokens:
            if self.signal_handler.stop:
                logging.warning("Stop requested; exiting before starting new shard.")
                break

            shard_capacity = min(
                self.cfg.shard_size_tokens,
                total_effective_tokens - pbar.n,
            )
            shard_num_seqs = shard_capacity // self.cfg.sequence_length
            shard_num_tokens = shard_num_seqs * self.cfg.sequence_length
            if shard_num_tokens == 0:
                break

            shard_dir, token_mm, act_mms = self.writer.prepare_memmaps(
                shard_index=shard_index,
                token_start=global_token_cursor,
                num_tokens=shard_num_tokens,
            )

            seqs_written = 0
            token_write_cursor = 0

            try:
                while seqs_written < shard_num_seqs:
                    batch_np = []
                    for _ in range(min(self.cfg.batch_size, shard_num_seqs - seqs_written)):
                        try:
                            batch_np.append(next(token_stream))
                        except StopIteration:
                            break

                    if not batch_np:
                        break

                    batch_np = np.stack(batch_np, axis=0)   # [B, S]
                    batch_tokens = torch.from_numpy(batch_np).to(self.cfg.device)

                    # Store tokens flattened.
                    flat_tokens = batch_np.reshape(-1)
                    n_batch_tokens = flat_tokens.shape[0]
                    token_mm[token_write_cursor : token_write_cursor + n_batch_tokens] = flat_tokens

                    # Extract aligned activations for each model over the exact same token batch.
                    for model_name, extractor in self.extractors.items():
                        acts = extractor.extract_batch(batch_tokens)  # [B, S, D] on CPU
                        acts_np = acts.reshape(-1, acts.shape[-1]).numpy()
                        act_mms[model_name][
                            token_write_cursor : token_write_cursor + n_batch_tokens
                        ] = acts_np

                    token_write_cursor += n_batch_tokens
                    seqs_written += batch_np.shape[0]
                    pbar.update(n_batch_tokens)

                # Flush memmaps
                token_mm.flush()
                for mm in act_mms.values():
                    mm.flush()

                actual_tokens = token_write_cursor
                if actual_tokens == 0:
                    logging.warning("Shard %d produced 0 tokens; removing empty shard.", shard_index)
                    try:
                        for fp in shard_dir.iterdir():
                            fp.unlink()
                        shard_dir.rmdir()
                    except Exception:
                        logging.exception("Failed to clean empty shard directory: %s", shard_dir)
                    break

                meta = ShardMeta(
                    shard_index=shard_index,
                    token_start=global_token_cursor,
                    token_end=global_token_cursor + actual_tokens,
                    num_tokens=actual_tokens,
                    hook_point=self.cfg.hook_point,
                    dtype=self.cfg.dtype,
                    token_dtype=self.cfg.token_dtype,
                    sequence_length=self.cfg.sequence_length,
                    batch_size=self.cfg.batch_size,
                    models=[asdict(m) for m in self.model_specs],
                )
                self.writer.commit_shard(meta, shard_dir)

                logging.info(
                    "Committed shard=%d token_range=[%d, %d)",
                    shard_index,
                    meta.token_start,
                    meta.token_end,
                )

                global_token_cursor += actual_tokens
                shard_index += 1

                if actual_tokens < shard_num_tokens:
                    logging.warning("Source stream ended early.")
                    break

            except Exception:
                logging.exception("Error while processing shard %d. Partial shard left uncommitted.", shard_index)
                raise

        pbar.close()
        logging.info("Done. Total committed tokens now: %d", global_token_cursor)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute aligned LLM activations for SAE training.")
    p.add_argument("--model-names", nargs="+", required=True)
    p.add_argument("--hook-point", type=str, required=True)
    p.add_argument("--num-tokens", type=int, required=True)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size-tokens", type=int, default=262144)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--token-dtype", type=str, default="int32", choices=["int32", "int64"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf-dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--hf-dataset-config", type=str, default="sample-100BT")
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--add-bos", action="store_true")
    p.add_argument("--no-append-eos-between-docs", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = PreprocessConfig(
        model_names=args.model_names,
        hook_point=args.hook_point,
        num_tokens=args.num_tokens,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        shard_size_tokens=args.shard_size_tokens,
        output_dir=args.output_dir,
        dtype=args.dtype,
        token_dtype=args.token_dtype,
        num_workers=args.num_workers,
        device=args.device,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        streaming=not args.no_streaming,
        add_bos=args.add_bos,
        append_eos_between_docs=not args.no_append_eos_between_docs,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    ActivationPreprocessor(cfg).run()


if __name__ == "__main__":
    main()