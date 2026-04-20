import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class AggregationGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, streams_z: Dict[str, torch.Tensor]):
        """
        return : mask of shape [B, latent_dim] with values in [0, 1] indicating how much to use each latent dimension
        """
        pass # To be implemented by subclasses



class TopKGate(AggregationGate):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        
    def forward(self, streams_z: Dict[str, torch.Tensor]):
        # streams_z: Dict[stream_name, [B, latent_dim]]
        # Stack into [B, num_streams, latent_dim]
        stream_names = list(streams_z.keys())
        z_stack = torch.stack([streams_z[name] for name in stream_names], dim=1)  # [B, S, D]
        z_sum = z_stack.sum(dim=1)  # [B, D]


        _, topk_indices = torch.topk(z_sum, self.k, dim=1)  # [B, k]
        mask = torch.zeros_like(z_sum, device=z_sum.device)  # [B, D]
        mask.scatter_(1, topk_indices, 1.0)  # Set 1 to the top-k indices
        return mask  # [B, D]


class SPARC(nn.Module):
    def __init__(self, streams : Dict[str, int], latent_dim: int, gate: AggregationGate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streams_dim = streams
        self.latent_dim = latent_dim
        self.gate = gate
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        self.pre_biases = nn.ParameterDict()
        self.latent_biases = nn.ParameterDict()

        for name, dim in streams.items():
            self.encoders[name] = nn.Linear(dim, latent_dim, bias=False)
            self.decoders[name] = nn.Linear(latent_dim, dim, bias=False)

            self.decoders[name].weight.data = self.encoders[name].weight.data.T.clone()

            self.pre_biases[name] = nn.Parameter(torch.zeros(dim))
            self.latent_biases[name] = nn.Parameter(torch.zeros(latent_dim))

    def _encode_stream(self, x : torch.Tensor, name: str):
        centered_x = x - self.pre_biases[name]
        return F.linear(centered_x, self.encoders[name].weight, self.latent_biases[name])

    def _decode_stream(self, z: torch.Tensor, name: str):
        return F.linear(z, self.decoders[name].weight, self.pre_biases[name])

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # Encode each stream
        streams_enc = {}
        for name, x in inputs.items():
            streams_enc[name] = self._encode_stream(x, name)

        # Get gate mask
        shared_mask = self.gate(streams_enc)  # [B, D]
        
        streams_z = {}
        # Apply mask and decode each stream
        outputs = {}
        for name in inputs.keys():
            streams_z[name] = streams_enc[name] * shared_mask  # [B, D]
            outputs[name] = self._decode_stream(streams_z[name], name)

        for name_z in inputs.keys():
            for name_dec in inputs.keys():
                if name_z != name_dec:
                    # Cross-decode
                    cross_decoded = self._decode_stream(streams_z[name_z], name_dec)
                    outputs["cross_" + name_z + "_from_" + name_dec] = cross_decoded

        return outputs, streams_z
