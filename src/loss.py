import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict



def nmse_loss(recon: torch.Tensor, target: torch.Tensor):
    mse = F.mse_loss(recon, target)
    var = target.var(dim=0).mean()
    nmse = mse / (var + 1e-8)
    return nmse

class SPARCLoss(nn.Module):
    def __init__(self, recon_weight: float = 1.0, cross_recon_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.cross_recon_weight = cross_recon_weight

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        loss = 0.0
        total_self_loss = 0.0
        total_cross_loss = 0.0
        self_loss = {}
        cross_loss = {}
        for name in outputs.keys():
            if name.startswith("cross_"):
                target_name = name.split("_from_")[1]
                l = nmse_loss(outputs[name], targets[target_name])
                cross_loss[name] = l.item()
                total_cross_loss += l
            else:
                l = nmse_loss(outputs[name], targets[name])
                self_loss[name] = l.item()
                total_self_loss += l

        loss = self.recon_weight * total_self_loss + self.cross_recon_weight * total_cross_loss
        return loss, self_loss, cross_loss