"""Evaluate a saved SPARC checkpoint on a streaming validation slice.

Loads the checkpoint, infers stream dims from encoder shapes, streams
~num-eval-positions aligned positions from FineWeb-Edu through the SAE, and
prints FVE/FVU per self-recon and cross-recon output.

Usage:
    python -m src.eval_checkpoint \
      --ckpt /ephemeral/spotai_small/dognev/unisae_big_ckpt.pt \
      --model-names Qwen/Qwen2.5-1.5B meta-llama/Llama-3.2-1B allenai/OLMo-1B-hf \
      --hook-points blocks.27.hook_resid_post blocks.15.hook_resid_post blocks.15.hook_resid_post \
      --k 128 --auxk 512 \
      --num-eval-positions 65536
"""

import argparse

import torch
from tqdm import tqdm

from src.SPARC import SPARC
from src.sparc_train import compute_fvu
from src.streaming_dataset import StreamingAlignedLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-names", nargs="+", required=True)
    ap.add_argument("--hook-points", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--auxk", type=int, default=0)
    ap.add_argument("--primary-sequence-length", type=int, default=512)
    ap.add_argument("--chunk-batch-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-eval-positions", type=int, default=65536)
    ap.add_argument("--prepend-bos", action="store_true", default=True)
    ap.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--hf-dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf-dataset-config", type=str, default="sample-100BT")
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--max-window", type=int, default=16)
    args = ap.parse_args()

    if len(args.hook_points) == 1 and len(args.model_names) > 1:
        args.hook_points = args.hook_points * len(args.model_names)
    if len(args.hook_points) != len(args.model_names):
        raise ValueError("--hook-points must match --model-names in length (or be exactly one)")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    # Detect factorized variant from state_dict.
    factorize_rank = None
    if "encoder_shared.weight" in sd:
        latent_dim_from_shared, factorize_rank = sd["encoder_shared.weight"].shape
        factorize_rank = int(factorize_rank)
        print(f"Detected factorized checkpoint with rank={factorize_rank}")

    # Infer stream dims and latent_dim from encoder shapes.
    streams = {}
    inferred_latent_dim = None
    for name in args.model_names:
        slug = SPARC._slugify(name)
        if factorize_rank is not None:
            key = f"encoder_in.{slug}.weight"
            available_prefix = "encoder_in."
            if key not in sd:
                available = [k for k in sd if k.startswith(available_prefix)]
                raise ValueError(f"No encoder_in for '{name}' (slug '{slug}') in checkpoint.\nAvailable: {available}")
            _, dim_in = sd[key].shape
            streams[name] = int(dim_in)
            inferred_latent_dim = int(latent_dim_from_shared)
        else:
            key = f"encoders.{slug}.weight"
            if key not in sd:
                available = [k for k in sd if k.startswith("encoders.")]
                raise ValueError(f"No encoder for '{name}' (slug '{slug}') in checkpoint.\nAvailable: {available}")
            latent_dim, dim_in = sd[key].shape
            streams[name] = int(dim_in)
            if inferred_latent_dim is None:
                inferred_latent_dim = int(latent_dim)
            elif int(latent_dim) != inferred_latent_dim:
                raise ValueError(f"Inconsistent latent_dim across streams: {latent_dim} vs {inferred_latent_dim}")

    print(f"Inferred streams (name -> hidden dim): {streams}")
    print(f"Inferred latent_dim: {inferred_latent_dim}")

    model = SPARC(
        streams=streams,
        latent_dim=inferred_latent_dim,
        k=args.k,
        auxk=args.auxk if args.auxk > 0 else None,
        factorize_rank=factorize_rank,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"load_state_dict: missing={missing}, unexpected={unexpected}")
    model = model.to(args.device)
    model.eval()

    n_batches = (args.num_eval_positions + args.batch_size - 1) // args.batch_size
    print(f"Streaming {n_batches} batches × {args.batch_size} = ~{n_batches * args.batch_size} positions...")

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

    class _ProgressLoader:
        """Wraps the streaming loader so compute_fvu's `for batch in loader` shows a progress bar."""
        def __init__(self, inner):
            self._inner = inner
        def __iter__(self):
            return iter(tqdm(self._inner, total=len(self._inner), desc="eval batches", unit="batch"))
        def __len__(self):
            return len(self._inner)

    results = compute_fvu(_ProgressLoader(loader), model, args.device)

    print()
    print("Validation FVE / FVU (this is over the first num-eval-positions of the stream — same slice the training run used):")
    self_keys = sorted(k for k in results if k.startswith("recon_"))
    cross_keys = sorted(k for k in results if k.startswith("cross_recon_"))
    for k in self_keys:
        m = results[k]
        print(f"  SELF  {k:55s} FVE={m['fve']:+.4f}  FVU={m['fvu']:.4f}")
    for k in cross_keys:
        m = results[k]
        print(f"  CROSS {k:55s} FVE={m['fve']:+.4f}  FVU={m['fvu']:.4f}")


if __name__ == "__main__":
    main()
