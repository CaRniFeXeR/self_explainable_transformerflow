import torch
import numpy as np

from .hungarian_matching import caculate_minimized_matched_cost


def hungarian_points_to_polygon(gt_points_filtered: torch.Tensor, predicted_points: torch.Tensor) -> torch.Tensor:

    # gt_points_filtered = remove_padded_values_from_points(gt_points)
    n_gt_points = len(gt_points_filtered)
    n_pred_points = len(predicted_points)


    if n_pred_points > n_gt_points:
        for i in range(0, n_pred_points - n_gt_points):
            index = i % n_gt_points
            gt_points_filtered = torch.cat((gt_points_filtered, gt_points_filtered[index].unsqueeze(0)))

    # create cost matrix shape ( 2 * (n_gt_points) + n_duplicated_gt_points, n_points_prediction)
    cost_matrix = torch.ones(len(gt_points_filtered), n_pred_points) * np.inf

    # for each real gt point add distance to each predicted points
    for i, gt_point in enumerate(gt_points_filtered):
        # se_points = torch.sum(torch.pow(predicted_points - gt_point, 2), dim=-1)
        se_points = torch.sum(torch.abs(predicted_points - gt_point), dim = -1)
        cost_matrix[i] = se_points

    # calculate matching
    summed_loss = caculate_minimized_matched_cost(cost_matrix)

    return summed_loss