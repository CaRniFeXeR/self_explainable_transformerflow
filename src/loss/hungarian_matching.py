from typing import Optional
from scipy.optimize import linear_sum_assignment
import torch


def match_pairs(cost_matrix: torch.FloatTensor , offset_matrix : Optional[torch.FloatTensor] = None):
    cost_matrix_numpy = cost_matrix.detach().numpy()

    if offset_matrix is not None:
        cost_matrix_numpy = cost_matrix_numpy + offset_matrix.detach().numpy()

    row_idx, col_idx = linear_sum_assignment(cost_matrix_numpy)

    return cost_matrix[row_idx, col_idx]


def caculate_minimized_matched_cost(cost_matrix: torch.FloatTensor, offset_matrix : Optional[torch.FloatTensor] = None):
    return torch.mean(match_pairs(cost_matrix, offset_matrix))
