from ..datastructures.configs.modelparams import ModelParams
from .FNN import FNN
from .Decoder import PerceiverDecoder
from .Encoder import ISABEncoder
from typing import Optional, Tuple
import torch
import torch.nn as nn



class FlowGATR(nn.Module):
    """
    Flow GAting TRansformer GATR
    Input: FlowCytomerty events + padding mask
    Output: gate x and y marker predictions and gate polygon prediction
    """

    def __init__(self, params: ModelParams) -> None:

        super(FlowGATR, self).__init__()
        self.n_polygon_out = params.n_polygon_out
        self.n_obj_queries = params.n_obj_queries
        self.hidden_dim = params.n_polygon_out * params.n_transformer_out_dim_multi
        self.dim_latent = params.dim_latent
        self.params = params
        self.query_embed = nn.Embedding(int(params.n_obj_queries * params.n_polygon_out / params.points_per_query), self.dim_latent)

        # transformer
        self.encoder = ISABEncoder(dim_input=params.dim_input, n_hidden_layers_ISAB=params.n_hidden_layers_ISAB, dim_latent=params.dim_latent)
        self.decoder = PerceiverDecoder(dim_latent=params.dim_latent, dim_output=self.dim_latent, n_hidden_layers=params.n_hidden_layers_decoder, n_perciver_blocks=params.n_perciever_blocks_decoder, num_heads_cross_attention=params.n_decoder_cross_att_heads)

        # polygon predictions
        self.polygon_embed = FNN(dim_input=self.dim_latent, dim_hidden=self.dim_latent * 2, dim_output=self.params.points_per_query * 2, n_hidden_layers=params.n_hidden_layers_polygon_out)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape (_, padded_event_length, dim_latent)
        encoded = self.encoder(x, padding_mask)

        # shape(_, num_obj_queries, hidden_dim)
        transformer_out = self.decoder(self.query_embed.weight.unsqueeze(0), encoded, padding_mask)

        return self._forward_pass_head(transformer_out)

    def _forward_pass_head(self, transformer_out: torch.Tensor) -> torch.Tensor:
        bsz = transformer_out.shape[0]

        polygons_pred = self.polygon_embed(transformer_out)
        polygons_pred = torch.reshape(polygons_pred, (bsz, self.n_obj_queries, self.n_polygon_out, 2))

        return polygons_pred