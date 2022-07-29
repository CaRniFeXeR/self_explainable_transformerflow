from dataclasses import dataclass


@dataclass
class ModelParams:
    dim_input: int
    n_hidden_layers_ISAB: int
    n_obj_queries: int
    n_hidden_layers_decoder: int
    dim_latent: int
    n_polygon_out: int
    n_decoder_cross_att_heads: int
    n_perciever_blocks_decoder : int = 0
    n_transformer_out_dim_multi: int = 4
    n_hidden_layers_polygon_out: int = 2
    points_per_query : int = -1
