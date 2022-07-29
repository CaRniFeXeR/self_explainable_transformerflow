import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    Simple Fully Connected Pytorch MLP with GELU activation and optional LayerNormalization
    dim_input:   dimensionality of input
    dim_hidden:             list of dims for hidden layer
    dim_output:             dim of output layer
    """

    def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, n_hidden_layers: int, layernorm: bool = False, **kwargs):
        super().__init__()

        if n_hidden_layers < 1:
            raise ValueError(f"'n_hidden_layers' must be at least 1, given value: '{n_hidden_layers}'")

        lin_layers = [nn.Linear(dim_input, dim_hidden)]
        if layernorm:
            lin_layers.append(nn.LayerNorm(dim_hidden))

        lin_layers.append(nn.GELU())

        for _ in range(1, n_hidden_layers):
            hidden_layer = [nn.Linear(dim_hidden, dim_hidden)]
            if layernorm:
                hidden_layer.append(nn.LayerNorm(dim_hidden))

            hidden_layer.append(nn.GELU())

            lin_layers.extend(hidden_layer)

        self.model = nn.Sequential(*lin_layers,
                                   nn.Linear(dim_hidden, dim_output)
                                   )

    def forward(self, input: torch.Tensor):

        return self.model(input)