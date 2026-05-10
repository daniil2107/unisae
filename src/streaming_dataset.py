"""Streaming aligned-activation loader for SPARC training.

Yields collated batches in the same shape as `dataset.AlignedActivationDataset`,
but generates them on the fly from FineWeb-Edu text — no shard files needed.

For each batch:
  - pull `chunk_batch_size` paragraph-packed text chunks from the stream
  - per-model tokenize + batched forward
  - run `align_n_models` per chunk
  - buffer aligned positions, yield once we have `batch_size` of them

A single underlying chunker is shared across `__iter__` calls, so successive
epochs continue forward through the dataset rather than restarting.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Optional

import numpy as np
import torch
from datasets import load_dataset

from src.dataset import collate_aligned_activation_batch
from src.precompute_activations import ActivationExtractor
from src.tokenizer_align import align_n_models, default_is_non_content


_PARA_RE = re.compile(r"\n+")


def _iter_text(name: str, config: str, split: str) -> Iterator[str]:
    ds = load_dataset(name, config, split=split, streaming=True)
    for row in ds:
        text = row.get("text", "")
        if text:
            yield text


def _iter_chunks(text_iter: Iterator[str], primary_tokenizer, seq_len: int) -> Iterator[str]:
    """Mirror of TextChunker in src.precompute_aligned, replicated here to
    avoid pulling that module's transformer_lens import chain twice."""
    para_sep_n = len(primary_tokenizer.encode("\n", add_special_tokens=False))
    for doc in text_iter:
        paragraphs = [p.strip() for p in _PARA_RE.split(doc) if p.strip()]
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


class StreamingAlignedLoader:
    def __init__(
        self,
        model_names: List[str],
        hook_points: List[str],
        batch_size: int,
        primary_sequence_length: int = 512,
        chunk_batch_size: int = 8,
        max_window: int = 16,
        prepend_bos: bool = True,
        device: str = "cuda",
        precision: str = "float16",
        hf_dataset_name: str = "HuggingFaceFW/fineweb-edu",
        hf_dataset_config: str = "sample-100BT",
        hf_split: str = "train",
        steps_per_epoch: Optional[int] = None,
    ):
        if len(model_names) != len(hook_points):
            raise ValueError("model_names and hook_points must have the same length")
        self.model_names = list(model_names)
        self.batch_size = batch_size
        self.primary_sequence_length = primary_sequence_length
        self.chunk_batch_size = chunk_batch_size
        self.max_window = max_window
        self.prepend_bos = prepend_bos
        self.device = device
        self.steps_per_epoch = steps_per_epoch

        self.extractors = {
            name: ActivationExtractor(model_name=name, hook_point=hp, device=device, precision=precision)
            for name, hp in zip(self.model_names, hook_points)
        }
        self.tokenizers = [self.extractors[n].model.tokenizer for n in self.model_names]
        self._chunker = _iter_chunks(
            _iter_text(hf_dataset_name, hf_dataset_config, hf_split),
            self.tokenizers[0],
            self.primary_sequence_length,
        )
        self._position_iter = self._produce_positions()
        self._buffer: list = []

    @property
    def stream_dims(self):
        return {name: self.extractors[name].hidden_size for name in self.model_names}

    def _forward_batch(self, model_name: str, token_lists: List[List[int]]) -> np.ndarray:
        extractor = self.extractors[model_name]
        tokenizer = extractor.model.tokenizer
        pad_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or 0
        )
        max_len = max((len(t) for t in token_lists), default=0)
        if max_len == 0:
            return np.empty((0, 0, extractor.hidden_size), dtype=np.float32)
        padded = np.full((len(token_lists), max_len), pad_id, dtype=np.int64)
        for i, t in enumerate(token_lists):
            padded[i, : len(t)] = t
        batch_tokens = torch.from_numpy(padded).to(self.device)
        acts = extractor.extract_batch(batch_tokens)
        return acts.float().numpy()

    def _produce_positions(self) -> Iterator[dict]:
        is_non_content_fns = [default_is_non_content(tk) for tk in self.tokenizers]
        single_stream = len(self.model_names) == 1
        while True:
            chunk_texts = []
            for _ in range(self.chunk_batch_size):
                try:
                    chunk_texts.append(next(self._chunker))
                except StopIteration:
                    break
            if not chunk_texts:
                return

            per_model_tokens: List[List[List[int]]] = []
            for tk in self.tokenizers:
                tok_lists = []
                for text in chunk_texts:
                    ids = tk.encode(text, add_special_tokens=False)
                    if self.prepend_bos and getattr(tk, "bos_token_id", None) is not None:
                        ids = [tk.bos_token_id] + ids
                    tok_lists.append(ids)
                per_model_tokens.append(tok_lists)

            per_model_acts = [
                self._forward_batch(name, per_model_tokens[m_idx])
                for m_idx, name in enumerate(self.model_names)
            ]

            if single_stream:
                # Bypass cross-tokenizer alignment entirely. Walk every real
                # position, skip non-content tokens, yield directly.
                name = self.model_names[0]
                is_nc = is_non_content_fns[0]
                for b_idx in range(len(chunk_texts)):
                    toks = per_model_tokens[0][b_idx]
                    if not toks:
                        continue
                    acts = per_model_acts[0][b_idx, : len(toks), :]
                    for k, tok_id in enumerate(toks):
                        if is_nc(int(tok_id)):
                            continue
                        yield {
                            "activations": {
                                name: torch.from_numpy(np.asarray(acts[k]).copy())
                            },
                            "tokens": {
                                name: torch.tensor([int(tok_id)], dtype=torch.long)
                            },
                        }
                continue

            for b_idx in range(len(chunk_texts)):
                streams = []
                skip = False
                for m_idx in range(len(self.model_names)):
                    toks = per_model_tokens[m_idx][b_idx]
                    if not toks:
                        skip = True
                        break
                    acts = per_model_acts[m_idx][b_idx, : len(toks), :]
                    streams.append((toks, acts, self.tokenizers[m_idx]))
                if skip:
                    continue
                res = align_n_models(streams, max_window=self.max_window)
                n_aligned = res.aligned[0].shape[0]
                for k in range(n_aligned):
                    yield {
                        "activations": {
                            name: torch.from_numpy(np.asarray(res.aligned[m_idx][k]).copy())
                            for m_idx, name in enumerate(self.model_names)
                        },
                        "tokens": {
                            name: torch.tensor([int(res.last_tok[m_idx][k])], dtype=torch.long)
                            for m_idx, name in enumerate(self.model_names)
                        },
                    }

    def __iter__(self):
        steps = 0
        while self.steps_per_epoch is None or steps < self.steps_per_epoch:
            while len(self._buffer) < self.batch_size:
                try:
                    self._buffer.append(next(self._position_iter))
                except StopIteration:
                    break
            if len(self._buffer) < self.batch_size:
                return
            batch_items = self._buffer[: self.batch_size]
            self._buffer = self._buffer[self.batch_size :]
            yield collate_aligned_activation_batch(batch_items)
            steps += 1

    def __len__(self) -> int:
        if self.steps_per_epoch is None:
            raise TypeError("StreamingAlignedLoader has no len() unless steps_per_epoch is set")
        return self.steps_per_epoch


class _CachedLoader:
    def __init__(self, batches: list):
        self._batches = batches

    def __iter__(self):
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        return len(self._batches)


def collect_validation_set(loader: StreamingAlignedLoader, num_positions: int) -> _CachedLoader:
    """Drain `num_positions` aligned positions from `loader`, cache the resulting
    batches in CPU memory, and return an iterable that re-yields them each epoch.

    Advances the underlying chunker — subsequent training on `loader` continues
    from where validation left off, so train/valid streams are disjoint.
    """
    from tqdm import tqdm
    batches = []
    collected = 0
    saved_steps = loader.steps_per_epoch
    loader.steps_per_epoch = None  # unbounded for collection
    pbar = tqdm(total=num_positions, desc="valid set", unit="pos")
    try:
        for batch in loader:
            batches.append(batch)
            n_in_batch = next(iter(batch["activations"].values())).shape[0]
            collected += n_in_batch
            pbar.update(n_in_batch)
            if collected >= num_positions:
                break
    finally:
        pbar.close()
        loader.steps_per_epoch = saved_steps
    return _CachedLoader(batches)
