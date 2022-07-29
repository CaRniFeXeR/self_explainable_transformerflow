from typing import List, Optional
import torch
import torch.nn as nn
from .ISAB import ISAB


class ISABEncoder(nn.Module):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    Needs [N, set_size, c] = [batchsize, n_events, n_marker/channels]
    _num_markers/dim_input: dimensionality of input              (flowdata: number of markers)
    dim_hidden_lin:         dim of eventwise embedding
    n_hidden_layers_ISAB:          number of isab blocks
    dim_hidden_att:         dim of hidden space in attention blocks, 
                            must be divisible by num_heads i.e. dim_hidden%num_heads = 0
    num_heads:              number of attention heads
    num_inds:               number of induced points
    layer_norm/ln:          use layer norm true/false
    dim_output/dim_latent:  dim of latent space, dim of events as output of encoder 
    """

    def __init__(self,
                 dim_input: int,
                 n_hidden_layers_ISAB: int,
                 num_inds: int = 16,
                 dim_hidden_att: int = 32,
                 num_heads: int = 4,
                 dim_latent: int = 32,
                 layer_norm: bool = True, **kwargs):
        super().__init__()

        # parameters for attention blocks part

        # induced self attention blocks,
        # dim_hidden_att should be divisable by num_heads,
        # points are projected in dim_hidden_att/num_heads spaces
        self.att_layers = [ISAB(dim_input, dim_hidden_att,
                           num_heads, num_inds, ln=layer_norm)]
        d_idx = 0
        for _ in range(1,  n_hidden_layers_ISAB):
            self.att_layers.append(ISAB(dim_hidden_att, dim_hidden_att, num_heads, num_inds, ln=layer_norm))
            d_idx += 1
        # dim_output should be dividable by num_heads
        self.att_layers.append(ISAB(dim_hidden_att, dim_latent, 1, num_inds, ln=layer_norm))
        self.att_layers = nn.ModuleList(self.att_layers)

    def forward(self, x: torch.Tensor, padding: Optional[torch.Tensor]):
        output = x
        for att in self.att_layers:
            output = att(output, padding)
        return output
