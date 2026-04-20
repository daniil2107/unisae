from src.SPARC import AggregationGate, SPARC, TopKGate
from src.loss import SPARCLoss
import argparse
import torch
import torch.nn as nn
from src.dataset import make_activation_dataloader, split_shards_train_valid
from tqdm import tqdm
from typing import Dict

def compute_r2_per_dim(x, x_hat):
    ss_res = ((x - x_hat) ** 2).sum(dim=0)
    ss_tot = ((x - x.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2 = 1 - ss_res / ss_tot
    return r2


@torch.no_grad()
def compute_r2_per_dim(x, x_hat):
    ss_res = ((x - x_hat) ** 2).sum(dim=0)
    ss_tot = ((x - x.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2 = 1 - ss_res / ss_tot
    return r2


@torch.no_grad()
def compute_R2(valid_loader, model, device) -> Dict[str, float]:
    model.eval()
    r2_scores = {}
    counts = {}
    for batch in valid_loader:
        inputs = {name: batch["activations"][name].to(device, dtype=torch.float32, non_blocking=True) for name in batch["activations"].keys()}
        outputs, _ = model(inputs)
        for name, output_tensor in outputs.items():
            if name.startswith("cross_"):
                target_stream = name.split("_")[3]
                target_tensor = inputs[target_stream]
            else:
                target_tensor = inputs[name]
            r2 = compute_r2_per_dim(target_tensor, output_tensor)
            r2_scores[name] = r2_scores.get(name, 0) + r2.mean().item()
            counts[name] = counts.get(name, 0) + 1

    for name in r2_scores:
        r2_scores[name] /= counts[name]

    return r2_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
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
    args = ap.parse_args()

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
    model = SPARC(streams=stream_dims, latent_dim=args.latent_dim, gate=TopKGate(k=args.k)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = SPARCLoss(recon_weight=args.recon_weight, cross_recon_weight=args.cross_recon_weight)

    for epoch in tqdm(range(args.num_epochs), desc="Epochs", unit="epoch"):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False, unit="batch"):
            optimizer.zero_grad()
            inputs = {name: batch["activations"][name].to(args.device, dtype=torch.float32, non_blocking=True) for name in batch["activations"].keys()}
            outputs, _ = model(inputs)

            loss, self_loss, cross_loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()


        # Compute R2 score on validation set
        streams_r2 = compute_R2(valid_loader, model, args.device)
        str_r2 = " | ".join([f"{name}: {r2:.4f}" for name, r2 in streams_r2.items()])

        dead_features = model.get_dead_features()
        str_dead = " | ".join(f"{name}: {dead.numel()/model.latent_dim:.2%}" for name, dead in dead_features.items())

        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {epoch_loss:.4f} Cross Loss: {sum(cross_loss.values()):.4f} Self Loss: {sum(self_loss.values()):.4f}")
        print(f"Validation R2 scores: {str_r2}")
        print("Dead Features:", str_dead)
    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
    # Example call
    # python -m src.sparc_train --store-dir /path/to/activations --latent-dim 16384 --lr 3e-4 --num-epochs 10 --batch-size 4096 --num-workers 4 --recon-weight 1.0 --cross-recon-weight 1.0 --k 128 --device cuda