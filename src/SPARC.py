"""Multi-stream sparse autoencoder mirroring the SPARC reference.

Architecture follows https://github.com/AtlasAnalyticsLab/SPARC
(`sparc/model/model_global.py`), with one deliberate divergence: per-stream
input-RMS calibration (`input_scale_{name}` buffers, `calibrate()` method) so
that streams with different raw magnitudes can be compared inside the global
top-k gate without one stream dominating.

Outputs returned by `forward(inputs)`:
    outputs[f'recon_{name}']                — self-reconstruction in raw input space
    outputs[f'cross_recon_{tgt}_from_{src}']— cross-reconstruction in raw target space
    outputs[f'sparse_codes_{name}']         — [B, latent_dim] sparse codes (scaled space)
    outputs[f'mask_{name}']                 — [B, latent_dim] bool mask of active latents
    outputs['shared_mask'], outputs['shared_indices'] — global top-k gate output
    outputs[f'auxk_recon_scaled_{name}']    — auxk reconstruction (scaled space, only if auxk set)
    outputs[f'auxk_target_scaled_{name}']   — auxk target = scaled main residual (only if auxk set)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import normalized_mean_squared_error
from src.model_utils import unit_norm_decoder_


class TopK(nn.Module):
    """Top-k along last dim, returning (bool_mask, indices)."""

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(x, k=self.k, dim=-1)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, topk.indices, True)
        return mask, topk.indices


class SPARC(nn.Module):
    @staticmethod
    def _slugify(name: str) -> str:
        # nn.ModuleDict / register_buffer reject keys with '.', '/', '-', etc.
        out = name
        for ch in "./-":
            out = out.replace(ch, "_")
        return out

    def slug(self, name: str) -> str:
        """Public helper: translate a user-facing stream name to its internal slug."""
        return self._slug[name]

    def __init__(
        self,
        streams: Dict[str, int],
        latent_dim: int,
        k: int,
        auxk: Optional[int] = None,
        auxk_threshold: float = 1e-3,
        dead_steps_threshold: int = 1000,
        factorize_rank: Optional[int] = None,
    ):
        super().__init__()
        self.streams_dim = streams
        self.streams = list(streams.keys())
        self._slug = {name: SPARC._slugify(name) for name in self.streams}
        self.latent_dim = latent_dim
        self.k = k
        self.auxk = auxk
        self.auxk_threshold = auxk_threshold
        self.dead_steps_threshold = dead_steps_threshold
        self.factorize_rank = factorize_rank if (factorize_rank is not None and factorize_rank > 0) else None

        self.pre_biases = nn.ParameterDict()
        self.latent_biases = nn.ParameterDict()

        if self.factorize_rank is not None:
            # Shared inner factors (one each, not per-stream).
            self.encoder_shared = nn.Linear(self.factorize_rank, latent_dim, bias=False)
            self.decoder_shared = nn.Linear(latent_dim, self.factorize_rank, bias=False)
            self.decoder_shared.weight.data = self.encoder_shared.weight.data.T.clone()
            # Per-stream input/output projections.
            self.encoder_in = nn.ModuleDict()
            self.decoder_out = nn.ModuleDict()
        else:
            self.encoders = nn.ModuleDict()
            self.decoders = nn.ModuleDict()

        for name, dim in streams.items():
            slug = self._slug[name]
            if self.factorize_rank is not None:
                ein = nn.Linear(dim, self.factorize_rank, bias=False)
                dout = nn.Linear(self.factorize_rank, dim, bias=False)
                dout.weight.data = ein.weight.data.T.clone()
                self.encoder_in[slug] = ein
                self.decoder_out[slug] = dout
            else:
                enc = nn.Linear(dim, latent_dim, bias=False)
                dec = nn.Linear(latent_dim, dim, bias=False)
                dec.weight.data = enc.weight.data.T.clone()
                unit_norm_decoder_(dec)
                self.encoders[slug] = enc
                self.decoders[slug] = dec

            self.pre_biases[slug] = nn.Parameter(torch.zeros(dim))
            self.latent_biases[slug] = nn.Parameter(torch.zeros(latent_dim))

            self.register_buffer(f'last_non_zero_{slug}', torch.zeros(latent_dim, dtype=torch.long))
            self.register_buffer(f'input_scale_{slug}', torch.ones(()))

        self.gate = TopK(k=k)
        self.auxk_gate = TopK(k=auxk) if (auxk is not None and auxk > 0) else None

    def _encode_stream(self, x: torch.Tensor, name: str) -> torch.Tensor:
        slug = self._slug[name]
        centered = x - self.pre_biases[slug]
        if self.factorize_rank is not None:
            h = self.encoder_in[slug](centered)
            return F.linear(h, self.encoder_shared.weight, self.latent_biases[slug])
        return F.linear(centered, self.encoders[slug].weight, self.latent_biases[slug])

    def _decode_stream(self, z: torch.Tensor, name: str) -> torch.Tensor:
        slug = self._slug[name]
        if self.factorize_rank is not None:
            h = self.decoder_shared(z)
            return F.linear(h, self.decoder_out[slug].weight, self.pre_biases[slug])
        return F.linear(z, self.decoders[slug].weight, self.pre_biases[slug])

    def _auxk_mask_fn(self, x: torch.Tensor, last_non_zero: torch.Tensor) -> torch.Tensor:
        dead_mask = last_non_zero > self.dead_steps_threshold
        return x * dead_mask

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        scaled_inputs = {
            name: x * getattr(self, f'input_scale_{self._slug[name]}')
            for name, x in inputs.items()
        }

        outputs: Dict[str, torch.Tensor] = {}
        all_logits: Dict[str, torch.Tensor] = {}

        for name in inputs.keys():
            all_logits[name] = self._encode_stream(scaled_inputs[name], name)

        aggregated_logits = torch.stack(list(all_logits.values()), dim=0).sum(dim=0)
        shared_mask, shared_indices = self.gate(aggregated_logits)
        outputs['shared_mask'] = shared_mask
        outputs['shared_indices'] = shared_indices

        for name in inputs.keys():
            slug = self._slug[name]
            logits = all_logits[name]
            scale = getattr(self, f'input_scale_{slug}')
            B = logits.shape[0]

            batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand_as(shared_indices)
            values_at_shared = logits[batch_idx, shared_indices]
            activated_values = F.relu(values_at_shared)

            sparse_codes = torch.zeros_like(logits)
            sparse_codes.scatter_(-1, shared_indices, activated_values)
            outputs[f'sparse_codes_{name}'] = sparse_codes

            mask_stream = torch.zeros_like(logits, dtype=torch.bool)
            mask_stream.scatter_(-1, shared_indices, True)
            outputs[f'mask_{name}'] = mask_stream

            # Update dead-neuron tracking — only in training mode to avoid validation contamination.
            if self.training:
                last_non_zero = getattr(self, f'last_non_zero_{slug}')
                activated_batch = (sparse_codes > self.auxk_threshold)
                activated_latent = activated_batch.any(dim=0).to(last_non_zero.dtype)
                last_non_zero *= (1 - activated_latent)
                last_non_zero += 1

            recon_scaled = self._decode_stream(sparse_codes, name)
            outputs[f'recon_scaled_{name}'] = recon_scaled
            outputs[f'recon_{name}'] = recon_scaled / scale

            if self.auxk_gate is not None:
                last_non_zero = getattr(self, f'last_non_zero_{slug}')
                masked_logits = self._auxk_mask_fn(logits, last_non_zero)
                _, auxk_indices = self.auxk_gate(masked_logits)

                auxk_batch_idx = torch.arange(B, device=logits.device).unsqueeze(1).expand_as(auxk_indices)
                auxk_values_at = masked_logits[auxk_batch_idx, auxk_indices]
                auxk_activated = F.relu(auxk_values_at)

                auxk_codes = torch.zeros_like(logits)
                auxk_codes.scatter_(-1, auxk_indices, auxk_activated)
                outputs[f'auxk_sparse_codes_{name}'] = auxk_codes

                auxk_recon_scaled = self._decode_stream(auxk_codes, name)
                outputs[f'auxk_recon_scaled_{name}'] = auxk_recon_scaled

                pre_b = self.pre_biases[slug]
                outputs[f'auxk_target_scaled_{name}'] = (
                    scaled_inputs[name] - recon_scaled.detach() + pre_b.detach()
                )

        for src in inputs.keys():
            for tgt in inputs.keys():
                if src == tgt:
                    continue
                src_codes = outputs[f'sparse_codes_{src}']
                cross_scaled = self._decode_stream(src_codes, tgt)
                outputs[f'cross_recon_{tgt}_from_{src}'] = cross_scaled / getattr(self, f'input_scale_{self._slug[tgt]}')

        streams_z = {name: outputs[f'sparse_codes_{name}'] for name in inputs.keys()}
        return outputs, streams_z

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        auxk_coef: float = 1.0 / 32,
        cross_loss_coef: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        device = next(self.parameters()).device
        n_streams = len(self.streams)

        # Self-recon NMSE (raw space).
        total_self = torch.zeros((), device=device)
        for name in self.streams:
            if name not in inputs:
                continue
            l = normalized_mean_squared_error(outputs[f'recon_{name}'], inputs[name])
            metrics[f'mse_loss_{name}'] = l
            total_self = total_self + l
        avg_self = total_self / max(n_streams, 1)
        metrics['avg_mse_loss'] = avg_self

        total_loss = avg_self

        # AuxK NMSE (scaled space — both recon and target are pre-unscale).
        if self.auxk_gate is not None:
            total_auxk = torch.zeros((), device=device)
            n_auxk = 0
            for name in self.streams:
                ak_recon = outputs.get(f'auxk_recon_scaled_{name}')
                ak_target = outputs.get(f'auxk_target_scaled_{name}')
                if ak_recon is None or ak_target is None:
                    continue
                l = normalized_mean_squared_error(ak_recon, ak_target)
                metrics[f'auxk_loss_{name}'] = l
                total_auxk = total_auxk + l
                n_auxk += 1
            if n_auxk > 0:
                avg_auxk = total_auxk / n_auxk
                metrics['avg_auxk_loss'] = avg_auxk
                total_loss = total_loss + auxk_coef * avg_auxk

        # Cross-recon NMSE (raw space).
        total_cross = torch.zeros((), device=device)
        n_cross = 0
        for src in self.streams:
            for tgt in self.streams:
                if src == tgt:
                    continue
                key = f'cross_recon_{tgt}_from_{src}'
                if key not in outputs or tgt not in inputs:
                    continue
                l = normalized_mean_squared_error(outputs[key], inputs[tgt])
                metrics[f'cross_loss_{tgt}_from_{src}'] = l
                total_cross = total_cross + l
                n_cross += 1
        if n_cross > 0:
            avg_cross = total_cross / n_cross
            metrics['avg_cross_loss'] = avg_cross
            total_loss = total_loss + cross_loss_coef * avg_cross

        metrics['total_loss'] = total_loss
        return total_loss, metrics

    def get_dead_features(self, threshold: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Return per-stream tensor of indices of latents inactive for > threshold consecutive forwards."""
        if threshold is None:
            threshold = self.dead_steps_threshold
        return {
            name: (getattr(self, f'last_non_zero_{self._slug[name]}') > threshold).nonzero(as_tuple=True)[0]
            for name in self.streams
        }

    @torch.no_grad()
    def calibrate(self, loader, num_batches: int = 256) -> Dict[str, float]:
        """Set per-stream `input_scale_{name}` so that scaled activations have unit median RMS.

        Walks up to `num_batches` from `loader`, collects per-sample RMS per stream,
        and stores `1 / median_rms` into each buffer. Returns the scales for logging.
        """
        from tqdm import tqdm
        was_training = self.training
        self.eval()
        rms_chunks: Dict[str, list] = {name: [] for name in self.streams}
        pbar = tqdm(total=num_batches, desc="calibrate", unit="batch")
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            for name in self.streams:
                x = batch["activations"][name].to(dtype=torch.float32)
                rms = (x * x).mean(dim=-1).sqrt()
                rms_chunks[name].append(rms.detach().cpu())
            pbar.update(1)
        pbar.close()

        scales: Dict[str, float] = {}
        for name in self.streams:
            if not rms_chunks[name]:
                scales[name] = 1.0
                continue
            all_rms = torch.cat(rms_chunks[name])
            median_rms = float(all_rms.median().item())
            scale = 1.0 / max(median_rms, 1e-8)
            getattr(self, f'input_scale_{self._slug[name]}').fill_(scale)
            scales[name] = scale

        if was_training:
            self.train()
        return scales
