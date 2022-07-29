from typing import Optional
import torch
import torch.nn as nn
from .ISAB import MAB, SAB


class PerceiverBlock(nn.Module):

    def __init__(self,
                 dim_latent: int,
                 dim_output: int,
                 n_hidden_layers: int,
                 num_heads: int = 4,
                 num_heads_cross_attention: int = 4,
                 ln: bool = True,  **kwargs):
        super().__init__()

        # dim_hidden_att should be divisable by num_heads,
        # points are projected in dim_hidden_att/num_heads spaces
        self.att_layers = [MAB(dim_latent, dim_latent, dim_latent, num_heads_cross_attention, ln=ln)]
        d_idx = 0
        for _ in range(1,  n_hidden_layers):
            self.att_layers.append(SAB(dim_latent, dim_latent, num_heads, ln=ln))
            d_idx += 1
        self.att_layers.append(SAB(dim_latent, dim_latent, num_heads, ln=ln))

        self.att_layers = nn.ModuleList(self.att_layers)
        self.lin = nn.Linear(dim_latent, dim_output)

    def forward(self, object_queries: torch.Tensor, memory: torch.Tensor, padding_mask: Optional[torch.Tensor]):

        att_out = self.att_layers[0](object_queries, memory, padding_mask)[0]

        for i in range(1, len(self.att_layers)):
            att_out = self.att_layers[i](att_out)

        output = self.lin(att_out)
        return output
