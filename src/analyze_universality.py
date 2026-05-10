"""Universal-feature analysis for a trained 3-stream SPARC checkpoint.

Streams ~num-positions of activations through the SAE and computes per-latent
agreement metrics across the three streams: gate-selection rate, per-stream
firing rate, agreement rate (active in all streams given gate-selected),
mean-value, value correlation across stream pairs, pairwise Jaccard.

Outputs:
    {output_prefix}.json                       — full metrics dump
    {output_prefix}_firing_rate_{slug}.png     — per-stream firing-rate hist
    {output_prefix}_agreement_rate.png         — agreement-rate hist (alive)
    {output_prefix}_pairwise_jaccard.png       — Jaccard heatmap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.SPARC import SPARC
from src.streaming_dataset import StreamingAlignedLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-names", nargs="+", required=True)
    ap.add_argument("--hook-points", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--auxk", type=int, default=0)
    ap.add_argument("--num-positions", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--chunk-batch-size", type=int, default=4)
    ap.add_argument("--primary-sequence-length", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--prepend-bos", action="store_true", default=True)
    ap.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    ap.add_argument("--max-window", type=int, default=16)
    ap.add_argument("--hf-dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf-dataset-config", type=str, default="sample-100BT")
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--alive-firing-threshold", type=float, default=1e-4,
                    help="Latents with firing rate above this are considered 'alive' for that stream.")
    ap.add_argument("--universal-agreement-threshold", type=float, default=0.9,
                    help="Latents with agreement_rate above this (and alive) are 'universal'.")
    ap.add_argument("--output-prefix", type=str, default="/ephemeral/spotai_small/dognev/universality")
    args = ap.parse_args()

    if len(args.hook_points) == 1 and len(args.model_names) > 1:
        args.hook_points = args.hook_points * len(args.model_names)
    if len(args.hook_points) != len(args.model_names):
        raise ValueError("--hook-points must match --model-names in length (or be exactly one)")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    factorize_rank = None
    latent_dim_from_shared = None
    if "encoder_shared.weight" in sd:
        latent_dim_from_shared, factorize_rank = sd["encoder_shared.weight"].shape
        factorize_rank = int(factorize_rank)
        print(f"Detected factorized checkpoint with rank={factorize_rank}")

    streams_dict = {}
    inferred_latent_dim = None
    for name in args.model_names:
        slug = SPARC._slugify(name)
        if factorize_rank is not None:
            key = f"encoder_in.{slug}.weight"
            if key not in sd:
                raise ValueError(f"No encoder_in for '{name}' (slug '{slug}') in checkpoint.")
            _, dim_in = sd[key].shape
            streams_dict[name] = int(dim_in)
            inferred_latent_dim = int(latent_dim_from_shared)
        else:
            key = f"encoders.{slug}.weight"
            if key not in sd:
                raise ValueError(f"No encoder for '{name}' (slug '{slug}') in checkpoint.")
            latent_dim, dim_in = sd[key].shape
            streams_dict[name] = int(dim_in)
            if inferred_latent_dim is None:
                inferred_latent_dim = int(latent_dim)

    print(f"Streams (name -> hidden dim): {streams_dict}")
    print(f"Latent dim: {inferred_latent_dim}")

    model = SPARC(
        streams=streams_dict,
        latent_dim=inferred_latent_dim,
        k=args.k,
        auxk=args.auxk if args.auxk > 0 else None,
        factorize_rank=factorize_rank,
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(args.device).eval()

    n_streams = len(args.model_names)
    L = inferred_latent_dim
    pairs = [(i, j) for i in range(n_streams) for j in range(i + 1, n_streams)]
    n_pairs = len(pairs)

    n_batches = (args.num_positions + args.batch_size - 1) // args.batch_size
    print(f"Streaming {n_batches} batches × {args.batch_size} = ~{n_batches * args.batch_size} positions")

    loader = StreamingAlignedLoader(
        model_names=args.model_names,
        hook_points=args.hook_points,
        batch_size=args.batch_size,
        primary_sequence_length=args.primary_sequence_length,
        chunk_batch_size=args.chunk_batch_size,
        max_window=args.max_window,
        prepend_bos=args.prepend_bos,
        device=args.device,
        precision=args.precision,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        steps_per_epoch=n_batches,
    )

    device = args.device
    n_gate = torch.zeros(L, dtype=torch.int64, device=device)
    n_alive_per_stream = torch.zeros(n_streams, L, dtype=torch.int64, device=device)
    n_alive_all = torch.zeros(L, dtype=torch.int64, device=device)
    sum_value_per_stream = torch.zeros(n_streams, L, dtype=torch.float64, device=device)
    sum_sq_per_stream = torch.zeros(n_streams, L, dtype=torch.float64, device=device)
    sum_xy_per_pair = torch.zeros(n_pairs, L, dtype=torch.float64, device=device)
    intersect_per_pair = torch.zeros(n_pairs, dtype=torch.int64, device=device)
    union_per_pair = torch.zeros(n_pairs, dtype=torch.int64, device=device)
    L0_per_stream = torch.zeros(n_streams, dtype=torch.int64, device=device)

    total_positions = 0

    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc="analyze", unit="batch"):
            inputs = {
                name: batch["activations"][name].to(device, dtype=torch.float32, non_blocking=True)
                for name in batch["activations"].keys()
            }
            outputs, _ = model(inputs)
            B = next(iter(inputs.values())).shape[0]
            total_positions += B

            mask = outputs[f"mask_{args.model_names[0]}"]  # [B, L] bool
            n_gate += mask.long().sum(dim=0)

            sparse = []
            active = []
            for s, name in enumerate(args.model_names):
                sc = outputs[f"sparse_codes_{name}"].detach()  # [B, L] fp32
                sparse.append(sc)
                a = sc > 0
                active.append(a)
                n_alive_per_stream[s] += a.long().sum(dim=0)
                sum_value_per_stream[s] += sc.sum(dim=0).double()
                sum_sq_per_stream[s] += (sc * sc).sum(dim=0).double()
                L0_per_stream[s] += a.long().sum()

            all_active = active[0]
            for a in active[1:]:
                all_active = all_active & a
            n_alive_all += all_active.long().sum(dim=0)

            for p, (s1, s2) in enumerate(pairs):
                intersect_per_pair[p] += (active[s1] & active[s2]).sum()
                union_per_pair[p] += (active[s1] | active[s2]).sum()
                sum_xy_per_pair[p] += (sparse[s1] * sparse[s2]).sum(dim=0).double()

    # Move accumulators to CPU.
    n_gate = n_gate.cpu()
    n_alive_per_stream = n_alive_per_stream.cpu()
    n_alive_all = n_alive_all.cpu()
    sum_value_per_stream = sum_value_per_stream.cpu()
    sum_sq_per_stream = sum_sq_per_stream.cpu()
    sum_xy_per_pair = sum_xy_per_pair.cpu()
    intersect_per_pair = intersect_per_pair.cpu()
    union_per_pair = union_per_pair.cpu()
    L0_per_stream = L0_per_stream.cpu()

    N = total_positions
    print(f"\nAggregated over {N} positions")

    # Derived metrics.
    gate_rate = n_gate.float() / N
    firing_rate = n_alive_per_stream.float() / N
    agreement_rate = n_alive_all.float() / n_gate.clamp(min=1).float()
    mean_value_per_stream = (sum_value_per_stream / n_alive_per_stream.clamp(min=1).float()).float()
    L0_mean = (L0_per_stream.double() / N).float()
    jaccard_per_pair = intersect_per_pair.float() / union_per_pair.clamp(min=1).float()

    # Per-latent value correlation across pairs.
    n_gate_f = n_gate.clamp(min=1).double()
    mean_x = sum_value_per_stream / n_gate_f.unsqueeze(0)
    var_x = (sum_sq_per_stream / n_gate_f.unsqueeze(0) - mean_x * mean_x).clamp(min=0)
    value_corr_per_pair = torch.zeros(n_pairs, L, dtype=torch.float64)
    for p, (s1, s2) in enumerate(pairs):
        cov = sum_xy_per_pair[p] / n_gate_f - mean_x[s1] * mean_x[s2]
        denom = (var_x[s1] * var_x[s2]).sqrt().clamp(min=1e-12)
        value_corr_per_pair[p] = cov / denom

    # Regime classification.
    alive_per_stream_bool = firing_rate > args.alive_firing_threshold
    n_alive_streams = alive_per_stream_bool.long().sum(dim=0)
    universal_mask = (agreement_rate > args.universal_agreement_threshold) & (gate_rate > args.alive_firing_threshold)
    stream_specific_mask = n_alive_streams == 1
    dead_mask = n_alive_streams == 0
    ambiguous_mask = ~(universal_mask | stream_specific_mask | dead_mask)
    n_universal = int(universal_mask.sum())
    n_stream_specific = int(stream_specific_mask.sum())
    n_dead = int(dead_mask.sum())
    n_ambiguous = int(ambiguous_mask.sum())

    alive = ~dead_mask
    if alive.any():
        agree_alive_max = agreement_rate.clone()
        agree_alive_max[~alive] = -1.0
        top_universal_idx = torch.topk(agree_alive_max, min(20, int(alive.sum().item()))).indices
        agree_alive_min = agreement_rate.clone()
        agree_alive_min[~alive] = float("inf")
        top_specific_idx = torch.topk(
            agree_alive_min, min(20, int(alive.sum().item())), largest=False
        ).indices
    else:
        top_universal_idx = torch.tensor([], dtype=torch.long)
        top_specific_idx = torch.tensor([], dtype=torch.long)

    # Print stdout summary.
    print("\n=== SUMMARY ===")
    print(f"latent_dim={L}, k={args.k}, auxk={args.auxk}")
    print(f"\nL0 per stream (avg active features per position; ≤ k):")
    for s, name in enumerate(args.model_names):
        print(f"  {name:35s}: {L0_mean[s].item():7.2f}")
    print(f"\nPairwise Jaccard (active latents, over all positions × latents):")
    for p, (s1, s2) in enumerate(pairs):
        print(f"  {args.model_names[s1]:30s} ↔ {args.model_names[s2]:30s}: {jaccard_per_pair[p].item():.4f}")
    print(f"\nRegime counts (out of {L} latents):")
    print(f"  universal (agreement > {args.universal_agreement_threshold}, gate_rate > {args.alive_firing_threshold}): "
          f"{n_universal:5d} ({100 * n_universal / L:5.2f}%)")
    print(f"  stream-specific (alive in 1 stream only):                      "
          f"{n_stream_specific:5d} ({100 * n_stream_specific / L:5.2f}%)")
    print(f"  dead (firing < {args.alive_firing_threshold} in all streams):  "
          f"{n_dead:5d} ({100 * n_dead / L:5.2f}%)")
    print(f"  ambiguous (alive in 2+, agreement ≤ {args.universal_agreement_threshold}): "
          f"{n_ambiguous:5d} ({100 * n_ambiguous / L:5.2f}%)")

    pct_list = [0.05, 0.25, 0.5, 0.75, 0.95]
    if alive.any():
        agree_alive = agreement_rate[alive]
        ps = torch.quantile(agree_alive.double(), torch.tensor(pct_list, dtype=torch.float64)).tolist()
        print(f"\nAgreement rate (over alive latents): "
              f"mean={agree_alive.mean().item():.4f}  "
              f"P05={ps[0]:.4f} P25={ps[1]:.4f} P50={ps[2]:.4f} P75={ps[3]:.4f} P95={ps[4]:.4f}")
    else:
        ps = [0.0] * 5

    print(f"\nMean per-latent value correlation (over latents with both std > 0):")
    pair_corr_means = {}
    for p, (s1, s2) in enumerate(pairs):
        v = value_corr_per_pair[p]
        v_finite = v[torch.isfinite(v)]
        mean_corr = float(v_finite.mean().item()) if v_finite.numel() > 0 else 0.0
        median_corr = float(v_finite.median().item()) if v_finite.numel() > 0 else 0.0
        print(f"  {args.model_names[s1]:30s} ↔ {args.model_names[s2]:30s}: "
              f"mean={mean_corr:+.4f}  median={median_corr:+.4f}")
        pair_corr_means[f"{args.model_names[s1]}|{args.model_names[s2]}"] = {
            "mean": mean_corr, "median": median_corr
        }

    print("\nTop universal latent indices (highest agreement, alive):")
    for idx in top_universal_idx[:10].tolist():
        firing = [float(firing_rate[s, idx].item()) for s in range(n_streams)]
        print(f"  idx={idx:6d}  agreement={agreement_rate[idx].item():.4f}  "
              f"gate={gate_rate[idx].item():.4f}  firing={['%.4f' % f for f in firing]}")
    print("Top stream-specific (lowest agreement, alive):")
    for idx in top_specific_idx[:10].tolist():
        firing = [float(firing_rate[s, idx].item()) for s in range(n_streams)]
        print(f"  idx={idx:6d}  agreement={agreement_rate[idx].item():.4f}  "
              f"gate={gate_rate[idx].item():.4f}  firing={['%.4f' % f for f in firing]}")

    # Save JSON.
    out_json = Path(args.output_prefix + ".json")
    summary = {
        "n_positions": int(N),
        "model_names": list(args.model_names),
        "latent_dim": int(L),
        "k": int(args.k),
        "auxk": int(args.auxk),
        "L0_mean": L0_mean.tolist(),
        "jaccard_per_pair": {
            f"{args.model_names[s1]}|{args.model_names[s2]}": jaccard_per_pair[p].item()
            for p, (s1, s2) in enumerate(pairs)
        },
        "regime_counts": {
            "universal": n_universal,
            "stream_specific": n_stream_specific,
            "dead": n_dead,
            "ambiguous": n_ambiguous,
        },
        "agreement_rate_percentiles": dict(zip(["p05", "p25", "p50", "p75", "p95"], ps)),
        "agreement_rate_mean": float(agreement_rate[alive].mean().item()) if alive.any() else 0.0,
        "value_corr_per_pair": pair_corr_means,
        "top_universal_idx": top_universal_idx.tolist(),
        "top_specific_idx": top_specific_idx.tolist(),
        # Full per-latent arrays.
        "gate_rate": gate_rate.tolist(),
        "firing_rate_per_stream": firing_rate.tolist(),
        "agreement_rate": agreement_rate.tolist(),
        "mean_value_per_stream": mean_value_per_stream.tolist(),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved JSON: {out_json}")

    # Plots.
    for s, name in enumerate(args.model_names):
        slug = SPARC._slugify(name)
        rates = firing_rate[s].numpy()
        plt.figure(figsize=(7, 5))
        plt.hist(np.clip(rates, 1e-6, None), bins=np.logspace(-6, 0, 60))
        plt.xscale("log")
        plt.xlabel("firing rate per latent (log)")
        plt.ylabel("count")
        plt.title(f"Firing-rate distribution — {name}")
        path = Path(args.output_prefix + f"_firing_rate_{slug}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    if alive.any():
        plt.figure(figsize=(7, 5))
        plt.hist(agreement_rate[alive].numpy(), bins=50)
        plt.xlabel("agreement rate (P[active in all streams | gate-selected])")
        plt.ylabel("count of alive latents")
        plt.title("Agreement-rate distribution over alive latents")
        path = Path(args.output_prefix + "_agreement_rate.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    j_matrix = np.eye(n_streams)
    for p, (s1, s2) in enumerate(pairs):
        v = jaccard_per_pair[p].item()
        j_matrix[s1, s2] = v
        j_matrix[s2, s1] = v
    short_names = [n.split("/")[-1] for n in args.model_names]
    plt.figure(figsize=(6, 5))
    plt.imshow(j_matrix, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(n_streams), short_names, rotation=30, ha="right")
    plt.yticks(range(n_streams), short_names)
    plt.title("Pairwise Jaccard (active latents)")
    for i in range(n_streams):
        for j in range(n_streams):
            color = "white" if j_matrix[i, j] < 0.5 else "black"
            plt.text(j, i, f"{j_matrix[i, j]:.3f}", ha="center", va="center", color=color)
    path = Path(args.output_prefix + "_pairwise_jaccard.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
