from typing import Optional
import torch
import torch.nn as nn

from .Perceiver import PerceiverBlock

from .ISAB import ISAB, MAB, SAB


class ISABDecoder(nn.Module):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    Needs [N, set_size, c] = [batchsize, n_events, n_marker/channels]
    dim_input/dim_latent:   dimensionality of latenspace, encoder output
    dim_output:             dim of output, thus n_marker = dim_input of encoder
    n_hidden_layers_MAB:          number of mab blocks
    dim_hidden_att:         dim of hidden space in attention blocks, 
                            must be divisible by num_heads i.e. dim_hidden%num_heads = 0
    num_heads:              number of attention heads
    num_inds:               number of induced points
    layer_norm/ln:          use layer norm true/false
    """

    def __init__(self,
                 dim_latent: int,
                 dim_output: int,
                 n_hidden_layers: int,
                 num_heads: int = 4,
                 num_heads_cross_attention: int = 4,
                 num_inds_points: int = 16,
                 layer_norm: bool = True,  **kwargs):
        super().__init__()

        # nn.MultiheadAttention
        # induced self attention blocks,
        # dim_hidden_att should be divisable by num_heads,
        # points are projected in dim_hidden_att/num_heads spaces
        self.att_layers = [MAB(dim_latent, dim_latent, dim_latent, num_heads_cross_attention, ln=layer_norm)]
        d_idx = 0
        for _ in range(1,  n_hidden_layers):
            self.att_layers.append(ISAB(dim_latent, dim_latent, num_heads, num_inds=num_inds_points, ln=layer_norm))
            d_idx += 1
        self.att_layers.append(ISAB(dim_latent, dim_latent, num_heads, num_inds=num_inds_points, ln=layer_norm))

        self.att_layers = nn.ModuleList(self.att_layers)
        self.lin = nn.Linear(dim_latent, dim_output)

    def forward(self, object_queries: torch.Tensor, memory: torch.Tensor, padding_mask: Optional[torch.Tensor]):

        att_out, att_weight = self.att_layers[0](object_queries.repeat((memory.shape[0], 1, 1)), memory, padding_mask)

        for i in range(1, len(self.att_layers)):
            att_out = self.att_layers[i](att_out)

        output = self.lin(att_out)
        return output


class SABDecoder(nn.Module):

    def __init__(self,
                 dim_latent: int,
                 dim_output: int,
                 n_hidden_layers: int,
                 num_heads: int = 4,
                 num_heads_cross_attention: int = 4,
                 layer_norm: bool = True,  **kwargs):
        super().__init__()

        # nn.MultiheadAttention
        # induced self attention blocks,
        # dim_hidden_att should be divisable by num_heads,
        # points are projected in dim_hidden_att/num_heads spaces
        self.att_layers = [MAB(dim_latent, dim_latent, dim_latent, num_heads_cross_attention, ln=layer_norm)]
        d_idx = 0
        for _ in range(1,  n_hidden_layers):
            self.att_layers.append(SAB(dim_latent, dim_latent, num_heads, ln=layer_norm))
            d_idx += 1
        self.att_layers.append(SAB(dim_latent, dim_latent, num_heads, ln=layer_norm))

        self.att_layers = nn.ModuleList(self.att_layers)
        self.lin = nn.Linear(dim_latent, dim_output)

    def forward(self, object_queries: torch.Tensor, memory: torch.Tensor, padding_mask: Optional[torch.Tensor]):

        att_out = self.att_layers[0](object_queries.repeat((memory.shape[0], 1, 1)), memory, padding_mask)

        for i in range(1, len(self.att_layers)):
            att_out = self.att_layers[i](att_out)

        output = self.lin(att_out)
        return output


class PerceiverDecoder(nn.Module):

    def __init__(self,
                 dim_latent: int,
                 dim_output: int,
                 n_perciver_blocks : int,
                 n_hidden_layers: int,
                 num_heads: int = 4,
                 num_heads_cross_attention: int = 4,
                 ln: bool = True,  **kwargs):
        super().__init__()

        # nn.MultiheadAttention
        # induced self attention blocks,
        # dim_hidden_att should be divisable by num_heads,
        # points are projected in dim_hidden_att/num_heads spaces
        self.att_layers = []
        for _ in range(0,  n_perciver_blocks):
            self.att_layers.append(PerceiverBlock(dim_latent, dim_latent, n_hidden_layers= n_hidden_layers, num_heads= num_heads, num_heads_cross_attention=num_heads_cross_attention, ln=ln))

        self.att_layers = nn.ModuleList(self.att_layers)
        self.lin = nn.Linear(dim_latent, dim_output)

    def forward(self, object_queries: torch.Tensor, memory: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        # todo use ISAB after 1. layer? --> no
        att_out = self.att_layers[0](object_queries.repeat((memory.shape[0], 1, 1)), memory, padding_mask)

        for i in range(1, len(self.att_layers)):
            att_out = self.att_layers[i](att_out, memory, padding_mask)

        output = self.lin(att_out)
        return output
