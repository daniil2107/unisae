from src.SPARC import SPARC
from src.model_utils import unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
import argparse
import torch
import torch.nn as nn
from src.dataset import make_activation_dataloader, split_shards_train_valid
from src.streaming_dataset import StreamingAlignedLoader, collect_validation_set
from tqdm import tqdm
from typing import Dict

@torch.no_grad()
def compute_fvu(valid_loader, model, device) -> Dict[str, Dict[str, float]]:
    """Compute FVU and FVE per reconstruction output over the full validation set.

    Scores `recon_{name}` (self) and `cross_recon_{tgt}_from_{src}` outputs only;
    skips internal artifacts (sparse codes, masks, auxk targets, etc.).

    FVU = sum_i ||x_i - x_hat_i||^2 / sum_i ||x_i - mu||^2,  FVE = 1 - FVU,
    with mu the per-dim mean of the target stream over the entire validation set.
    Accumulators are kept in float64 to avoid cancellation in sum_x2 - N*mu^2.
    """
    model.eval()
    sum_sq_res: Dict[str, torch.Tensor] = {}
    sum_x: Dict[str, torch.Tensor] = {}
    sum_sq_x: Dict[str, torch.Tensor] = {}
    n: Dict[str, int] = {}

    for batch in valid_loader:
        inputs = {name: batch["activations"][name].to(device, dtype=torch.float32, non_blocking=True)
                  for name in batch["activations"].keys()}
        outputs, _ = model(inputs)

        scored: Dict[str, tuple] = {}
        for name in inputs.keys():
            recon_key = f'recon_{name}'
            if recon_key in outputs:
                scored[recon_key] = (outputs[recon_key], inputs[name])
        for src in inputs.keys():
            for tgt in inputs.keys():
                if src == tgt:
                    continue
                cross_key = f'cross_recon_{tgt}_from_{src}'
                if cross_key in outputs:
                    scored[cross_key] = (outputs[cross_key], inputs[tgt])

        for key, (output_tensor, target_tensor) in scored.items():
            res = (target_tensor - output_tensor).to(torch.float64)
            tgt = target_tensor.to(torch.float64)
            b_sq_res = (res * res).sum()
            b_sum_x = tgt.sum(dim=0)
            b_sum_sq_x = (tgt * tgt).sum(dim=0)

            if key not in sum_sq_res:
                sum_sq_res[key] = b_sq_res
                sum_x[key] = b_sum_x
                sum_sq_x[key] = b_sum_sq_x
                n[key] = target_tensor.shape[0]
            else:
                sum_sq_res[key] += b_sq_res
                sum_x[key] += b_sum_x
                sum_sq_x[key] += b_sum_sq_x
                n[key] += target_tensor.shape[0]

    results: Dict[str, Dict[str, float]] = {}
    for key in sum_sq_res:
        N = n[key]
        mu = sum_x[key] / N
        ss_tot = (sum_sq_x[key] - N * mu * mu).sum().clamp(min=1e-12)
        fvu = (sum_sq_res[key] / ss_tot).item()
        results[key] = {"fvu": fvu, "fve": 1.0 - fvu}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=None,
                    help="Aligned activation store dir; required unless --streaming.")
    ap.add_argument("--latent-dim", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--recon-weight", type=float, default=1.0)
    ap.add_argument("--cross-recon-weight", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save-path", type=str, default=None)
    ap.add_argument("--calib-batches", type=int, default=256, help="Batches used to calibrate per-stream input scales")
    # Streaming mode: forward + align on-the-fly, no shard files.
    ap.add_argument("--streaming", action="store_true",
                    help="Stream activations directly from FineWeb-Edu instead of reading from --store-dir.")
    ap.add_argument("--model-names", nargs="+", default=None,
                    help="Required with --streaming.")
    ap.add_argument("--hook-points", nargs="+", default=None,
                    help="One per model (or one to apply to all). Required with --streaming.")
    ap.add_argument("--primary-sequence-length", type=int, default=512)
    ap.add_argument("--chunk-batch-size", type=int, default=8,
                    help="Text chunks per forward pass when streaming.")
    ap.add_argument("--max-window", type=int, default=16)
    ap.add_argument("--prepend-bos", action="store_true", default=True)
    ap.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false")
    ap.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--hf-dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf-dataset-config", type=str, default="sample-100BT")
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--steps-per-epoch", type=int, default=4000,
                    help="Training batches per epoch in streaming mode.")
    ap.add_argument("--streaming-valid-positions", type=int, default=65536,
                    help="Positions held out for validation in streaming mode (0 = skip validation).")
    # SPARC mirror knobs.
    ap.add_argument("--auxk", type=int, default=0,
                    help="AuxK auxiliary top-k size (0 = disabled). Typical: 4 * k.")
    ap.add_argument("--auxk-coef", type=float, default=1.0 / 32,
                    help="Coefficient on AuxK auxiliary loss.")
    ap.add_argument("--auxk-threshold", type=float, default=1e-3,
                    help="Activation cutoff below which a latent counts as inactive.")
    ap.add_argument("--dead-neuron-threshold", type=int, default=1000,
                    help="Consecutive inactive forwards before a latent is flagged dead.")
    ap.add_argument("--reinit-every", type=int, default=100,
                    help="Run dead-neuron reinit every N training steps (0 = disabled).")
    ap.add_argument("--factorize-rank", type=int, default=0,
                    help="If >0, factorize per-stream encoder/decoder via a shared inner factor of this rank.")
    args = ap.parse_args()

    if args.streaming:
        if not args.model_names or not args.hook_points:
            raise ValueError("--streaming requires --model-names and --hook-points")
        if len(args.hook_points) == 1 and len(args.model_names) > 1:
            args.hook_points = args.hook_points * len(args.model_names)
        if len(args.hook_points) != len(args.model_names):
            raise ValueError("--hook-points must have the same length as --model-names (or exactly one)")
        train_loader = StreamingAlignedLoader(
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
            steps_per_epoch=args.steps_per_epoch,
        )
        if args.streaming_valid_positions > 0:
            print(f"Collecting {args.streaming_valid_positions} streaming validation positions...")
            valid_loader = collect_validation_set(train_loader, num_positions=args.streaming_valid_positions)
        else:
            valid_loader = None
        stream_dims = train_loader.stream_dims
    else:
        if not args.store_dir:
            raise ValueError("--store-dir is required when not streaming")
        _, _, train_loader, valid_loader = split_shards_train_valid(
            root=args.store_dir,
            valid_fraction=0.02,
            seed=42,
            model_names=None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        first_batch = next(iter(train_loader))
        stream_dims = {name: first_batch["activations"][name].shape[-1] for name in first_batch["activations"].keys()}
    model = SPARC(
        streams=stream_dims,
        latent_dim=args.latent_dim,
        k=args.k,
        auxk=args.auxk if args.auxk > 0 else None,
        auxk_threshold=args.auxk_threshold,
        dead_steps_threshold=args.dead_neuron_threshold,
        factorize_rank=args.factorize_rank if args.factorize_rank > 0 else None,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scales = model.calibrate(train_loader, num_batches=args.calib_batches)
    print("Calibrated input scales (unit RMS per stream):", {n: f"{s:.4f}" for n, s in scales.items()})

    global_step = 0
    last_metrics: Dict[str, torch.Tensor] = {}
    for epoch in tqdm(range(args.num_epochs), desc="Epochs", unit="epoch"):
        epoch_loss = 0.0
        num_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False, unit="batch"):
            optimizer.zero_grad()
            inputs = {name: batch["activations"][name].to(args.device, dtype=torch.float32, non_blocking=True) for name in batch["activations"].keys()}
            outputs, _ = model(inputs)

            loss, metrics = model.compute_loss(outputs, inputs, args.auxk_coef, args.cross_recon_weight)

            loss.backward()

            # Decoder unit-norm constraint: renormalize columns and project gradient onto the unit-sphere tangent.
            # Skipped in factorized mode — unit-norm of (decoder_shared @ decoder_out_s) does not decompose into per-factor norms.
            if model.factorize_rank is None:
                for name in model.streams:
                    dec = model.decoders[model.slug(name)]
                    unit_norm_decoder_(dec)
                    unit_norm_decoder_grad_adjustment_(dec)

            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            last_metrics = metrics

            # Dead-neuron reinit (matches SPARC reference, every reinit_every steps).
            if args.reinit_every > 0 and global_step % args.reinit_every == 0:
                dead_idx_dict = model.get_dead_features(threshold=args.dead_neuron_threshold)
                if any(idx.numel() > 0 for idx in dead_idx_dict.values()):
                    with torch.no_grad():
                        # In factorized mode the encoder is shared, so we reinit
                        # the dead rows ONCE on the shared factor (using the union
                        # of dead indices across streams) instead of per-stream.
                        if model.factorize_rank is not None:
                            all_dead = torch.unique(torch.cat([di for di in dead_idx_dict.values() if di.numel() > 0]))
                            if all_dead.numel() > 0:
                                es = model.encoder_shared.weight.data
                                es[all_dead] = torch.randn_like(es[all_dead]) * 0.01
                        for name, dead_idx in dead_idx_dict.items():
                            if dead_idx.numel() == 0:
                                continue
                            slug = model.slug(name)
                            if model.factorize_rank is None:
                                enc = model.encoders[slug]
                                enc.weight.data[dead_idx] = torch.randn_like(enc.weight.data[dead_idx]) * 0.01
                            model.latent_biases[slug].data[dead_idx] = 0
                            getattr(model, f'last_non_zero_{slug}')[dead_idx] = 0

        epoch_loss /= max(num_batches, 1)

        dead_features = model.get_dead_features()
        str_dead = " | ".join(f"{name}: {dead.numel()/model.latent_dim:.2%}" for name, dead in dead_features.items())

        avg_self = float(last_metrics.get('avg_mse_loss', torch.tensor(0.0)).item())
        avg_cross = float(last_metrics.get('avg_cross_loss', torch.tensor(0.0)).item()) if 'avg_cross_loss' in last_metrics else 0.0
        avg_auxk = float(last_metrics.get('avg_auxk_loss', torch.tensor(0.0)).item()) if 'avg_auxk_loss' in last_metrics else 0.0
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {epoch_loss:.4f} | Self: {avg_self:.4f} | Cross: {avg_cross:.4f} | AuxK: {avg_auxk:.4f}")
        if valid_loader is not None:
            streams_fvu = compute_fvu(valid_loader, model, args.device)
            str_fve = " | ".join(f"{name}: FVE={m['fve']:.4f} FVU={m['fvu']:.4f}" for name, m in streams_fvu.items())
            print(f"Validation: {str_fve}")
        print("Dead Features:", str_dead)
    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
    # Example call
    # python -m src.sparc_train --store-dir /path/to/activations --latent-dim 16384 --lr 3e-4 --num-epochs 10 --batch-size 4096 --num-workers 4 --recon-weight 1.0 --cross-recon-weight 1.0 --k 128 --device cuda