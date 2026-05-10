from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


@dataclass
class StoreConfig:
    root: Path
    dtype: str
    token_dtype: str
    batch_size: int
    shard_size_tokens: int
    models: List[dict]
    alignment_mode: str = "none"                    # "none" (shared tokenizer) or "greedy" (aligned)
    hook_point: Optional[str] = None                # present only when alignment_mode == "none"
    sequence_length: Optional[int] = None           # present only when alignment_mode == "none"
    primary_sequence_length: Optional[int] = None   # present only when alignment_mode == "greedy"


@dataclass
class ShardRecord:
    shard_index: int
    token_start: int
    token_end: int
    num_tokens: int
    path: Path


def load_store_config(root: str | Path) -> StoreConfig:
    root = Path(root)
    with (root / "store_config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return StoreConfig(
        root=root,
        dtype=cfg["dtype"],
        token_dtype=cfg["token_dtype"],
        batch_size=cfg["batch_size"],
        shard_size_tokens=cfg["shard_size_tokens"],
        models=cfg["models"],
        alignment_mode=cfg.get("alignment_mode", "none"),
        hook_point=cfg.get("hook_point"),
        sequence_length=cfg.get("sequence_length"),
        primary_sequence_length=cfg.get("primary_sequence_length"),
    )


def load_manifest(root: str | Path) -> List[ShardRecord]:
    root = Path(root)
    manifest_path = root / "manifest.jsonl"
    rows: List[ShardRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                ShardRecord(
                    shard_index=d["shard_index"],
                    token_start=d["token_start"],
                    token_end=d["token_end"],
                    num_tokens=d["num_tokens"],
                    path=root / "shards" / f"shard_{d['shard_index']:06d}",
                )
            )
    rows.sort(key=lambda x: x.shard_index)
    return rows


class _LazyShardCache:
    """
    Per-worker lazy memmap cache.
    Keeps only the currently needed shard open.

    In alignment_mode == "none", a single `tokens.npy` is loaded per shard.
    In alignment_mode == "greedy", per-model `last_tok_{slug}.npy` is loaded.
    """

    def __init__(self, store_cfg: StoreConfig):
        self.store_cfg = store_cfg
        self.current_shard_index: Optional[int] = None
        self.tokens_mm = None
        self.tokens_mms: Dict[str, np.memmap] = {}
        self.act_mms: Dict[str, np.memmap] = {}

    def open_shard(self, shard: ShardRecord):
        if self.current_shard_index == shard.shard_index:
            return

        self.act_mms = {}
        self.tokens_mms = {}
        self.tokens_mm = None
        for model in self.store_cfg.models:
            model_name = model["name"]
            slug = model["slug"]
            self.act_mms[model_name] = np.load(
                shard.path / f"model_{slug}.npy",
                mmap_mode="r",
            )

        if self.store_cfg.alignment_mode == "greedy":
            for model in self.store_cfg.models:
                self.tokens_mms[model["name"]] = np.load(
                    shard.path / f"last_tok_{model['slug']}.npy",
                    mmap_mode="r",
                )
        else:
            self.tokens_mm = np.load(shard.path / "tokens.npy", mmap_mode="r")

        self.current_shard_index = shard.shard_index

    def read_range(self, shard: ShardRecord, start: int, end: int):
        self.open_shard(shard)
        activations = {name: mm[start:end] for name, mm in self.act_mms.items()}
        if self.store_cfg.alignment_mode == "greedy":
            tokens = {name: mm[start:end] for name, mm in self.tokens_mms.items()}
        else:
            tokens = self.tokens_mm[start:end]
        return tokens, activations


class AlignedActivationDataset(Dataset):
    """
    Token-level dataset over a chosen subset of shards.
    Global dataset indices are remapped to local contiguous indices [0, len(self)).

    Each item:
        {
            "tokens": LongTensor[1],
            "activations": {
                "model_a": FloatTensor[D1],
                ...
            }
        }
    """

    def __init__(
        self,
        root: str | Path,
        model_names: Optional[List[str]] = None,
        shards: Optional[Sequence[ShardRecord]] = None,
    ):
        self.store_cfg = load_store_config(root)
        all_shards = load_manifest(root)
        if not all_shards:
            raise ValueError(f"No shards found in store: {root}")

        self.shards = list(shards) if shards is not None else all_shards
        if not self.shards:
            raise ValueError("Shard subset is empty.")

        available = [m["name"] for m in self.store_cfg.models]
        self.model_names = model_names or available
        for name in self.model_names:
            if name not in available:
                raise ValueError(f"Requested model '{name}' not found in store. Available={available}")

        self._cache = _LazyShardCache(self.store_cfg)

        # Local contiguous indexing for subset datasets.
        self._local_offsets: List[int] = []
        running = 0
        for shard in self.shards:
            self._local_offsets.append(running)
            running += shard.num_tokens
        self.total_tokens = running
        self._local_ends = [start + shard.num_tokens for start, shard in zip(self._local_offsets, self.shards)]

    def __len__(self) -> int:
        return self.total_tokens

    def _locate(self, index: int) -> Tuple[ShardRecord, int]:
        if index < 0 or index >= self.total_tokens:
            raise IndexError(index)

        lo, hi = 0, len(self._local_ends) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if index < self._local_ends[mid]:
                hi = mid
            else:
                lo = mid + 1

        shard = self.shards[lo]
        shard_local_base = self._local_offsets[lo]
        local_idx = index - shard_local_base
        return shard, local_idx

    def __getitem__(self, index: int):
        shard, local_idx = self._locate(index)
        tokens_np, acts_np = self._cache.read_range(shard, local_idx, local_idx + 1)

        item: dict = {"activations": {}}
        if isinstance(tokens_np, dict):
            # aligned mode: per-model last-token ids
            item["tokens"] = {
                name: torch.from_numpy(np.asarray(toks, dtype=np.int64).copy())
                for name, toks in tokens_np.items()
            }
        else:
            item["tokens"] = torch.from_numpy(np.asarray(tokens_np, dtype=np.int64).copy())
        for model_name in self.model_names:
            item["activations"][model_name] = torch.from_numpy(
                np.asarray(acts_np[model_name][0]).copy()
            )
        return item


class ShardContiguousBatchSampler(BatchSampler):
    """
    Batch sampler that yields contiguous token ranges within each dataset shard subset.
    """

    def __init__(
        self,
        dataset: AlignedActivationDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)

        shard_order = list(range(len(self.dataset.shards)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for shard_idx in shard_order:
            shard = self.dataset.shards[shard_idx]
            local_dataset_start = self.dataset._local_offsets[shard_idx]
            local_dataset_end = local_dataset_start + shard.num_tokens

            batch_starts = list(range(local_dataset_start, local_dataset_end, self.batch_size))
            if self.shuffle:
                rng.shuffle(batch_starts)

            for start in batch_starts:
                end = min(start + self.batch_size, local_dataset_end)
                if end - start < self.batch_size and self.drop_last:
                    continue
                yield list(range(start, end))

    def __len__(self):
        total = 0
        for shard in self.dataset.shards:
            n = shard.num_tokens // self.batch_size
            if not self.drop_last and shard.num_tokens % self.batch_size:
                n += 1
            total += n
        return total


def collate_aligned_activation_batch(batch: List[dict]) -> dict:
    first = batch[0]
    model_names = list(first["activations"].keys())
    activations = {
        name: torch.stack([x["activations"][name] for x in batch], dim=0)
        for name in model_names
    }
    if isinstance(first["tokens"], dict):
        tokens = {
            name: torch.cat([x["tokens"][name] for x in batch], dim=0)
            for name in first["tokens"]
        }
    else:
        tokens = torch.cat([x["tokens"] for x in batch], dim=0)
    return {"tokens": tokens, "activations": activations}


def make_activation_dataloader(
    root: str | Path,
    model_names: Optional[List[str]],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
    shards: Optional[Sequence[ShardRecord]] = None,
) -> DataLoader:
    dataset = AlignedActivationDataset(
        root=root,
        model_names=model_names,
        shards=shards,
    )

    batch_sampler = ShardContiguousBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        collate_fn=collate_aligned_activation_batch,
    )


def split_shards_train_valid(
    root: str | Path,
    valid_fraction: float = 0.01,
    seed: int = 42,
    model_names: Optional[List[str]] = None,
    batch_size: int = 4096,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
):
    """
    Split activation store into train/valid at the shard level.

    Returns:
        train_dataset, valid_dataset, train_loader, valid_loader

    Notes:
    - validation loader uses shuffle=False for deterministic evaluation
    - training loader uses shuffle=True
    - at least 1 shard is assigned to validation if there are >= 2 shards
    """
    all_shards = load_manifest(root)
    if len(all_shards) < 2:
        raise ValueError("Need at least 2 shards to create a train/valid split.")

    if not (0.0 < valid_fraction < 1.0):
        raise ValueError(f"valid_fraction must be in (0, 1), got {valid_fraction}")

    rng = random.Random(seed)
    shard_indices = list(range(len(all_shards)))
    rng.shuffle(shard_indices)

    n_valid = max(1, int(math.ceil(len(all_shards) * valid_fraction)))
    n_valid = min(n_valid, len(all_shards) - 1)

    valid_idx_set = set(shard_indices[:n_valid])

    train_shards = [sh for i, sh in enumerate(all_shards) if i not in valid_idx_set]
    valid_shards = [sh for i, sh in enumerate(all_shards) if i in valid_idx_set]

    train_dataset = AlignedActivationDataset(
        root=root,
        model_names=model_names,
        shards=train_shards,
    )
    valid_dataset = AlignedActivationDataset(
        root=root,
        model_names=model_names,
        shards=valid_shards,
    )

    train_loader = make_activation_dataloader(
        root=root,
        model_names=model_names,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
        shards=train_shards,
    )

    valid_loader = make_activation_dataloader(
        root=root,
        model_names=model_names,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=False,
        shards=valid_shards,
    )

    return train_dataset, valid_dataset, train_loader, valid_loader