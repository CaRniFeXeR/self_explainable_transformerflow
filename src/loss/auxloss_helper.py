from typing import List
import torch

from src.model.FlowGATR import FlowGATR


def register_perciever_forward_hock(intermed_result_list : List[torch.Tensor], model : FlowGATR):
    """
    registers forward hock
    """
    def hock_execution(self, input, output):
        if model.training:
            intermed_result_list.append(model._forward_pass_head(output))

    for preciever_block in model.decoder.att_layers[:-1]:
        preciever_block.register_forward_hook(hock_execution)

def compute_auxiliary_loss(intermed_result_list : List[torch.Tensor], gt_polygon : torch.Tensor, batch_idx : int,  loss_fn, increasing_weight : bool = True) -> torch.Tensor:
    auxiliary_losses = []
    n_layers = len(intermed_result_list)
    normalize_value = sum([1/n for n in range(2,n_layers + 1)])

    def increasing_weight_func(n_layers, i):
        return 1/(n_layers + 1 - i) * 1/normalize_value
    
    def equal_weight_func(n_layers, i):
        return 1/(n_layers) 

    weight_func = increasing_weight_func if increasing_weight else equal_weight_func

    for i in range(n_layers):
        intermed_res = intermed_result_list[i]
        current_layer_loss = torch.sum(loss_fn(intermed_res[batch_idx], gt_polygon[batch_idx]).point_loss)
        auxiliary_losses.append(weight_func(n_layers,i) * current_layer_loss)
    
    if auxiliary_losses == []:
        raise ValueError("no auxiliary loss collected")

    auxiliary_loss = torch.Tensor.sum(torch.stack(auxiliary_losses))
    return auxiliary_loss