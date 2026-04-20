#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import make_activation_dataloader


class SimpleSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.encoder(x))
        x_hat = self.decoder(z)
        return x_hat, z


def sae_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, l1_coeff: float):
    recon = F.mse_loss(x_hat, x)
    sparsity = z.abs().mean()
    return recon + l1_coeff * sparsity, recon.detach(), sparsity.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--latent-dim", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l1-coeff", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    loader = make_activation_dataloader(
        root=args.store_dir,
        model_names=[args.target_model],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    first_batch = next(iter(loader))
    input_dim = first_batch["activations"][args.target_model].shape[-1]

    model = SimpleSAE(input_dim=input_dim, latent_dim=args.latent_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    iterator = iter(loader)
    pbar = tqdm(range(args.steps), desc="sae_train")

    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        x = batch["activations"][args.target_model].to(
            args.device, dtype=torch.float32, non_blocking=True
        )
        x_hat, z = model(x)
        loss, recon, sparsity = sae_loss(x, x_hat, z, args.l1_coeff)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon.item():.4f}",
            sparsity=f"{sparsity.item():.4f}",
        )

    print("Training complete.")


if __name__ == "__main__":
    main()