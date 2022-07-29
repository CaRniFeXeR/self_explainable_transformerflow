import torch
from ..datastructures.LossResult import LossResult
from ..loss.loss_utils import remove_padded_values_from_points
from .hungarian_polygon_loss import hungarian_points_to_polygon


def flowsample_gating_loss(polygon_weight: float = 1.0):
    """
    returns the flow sample gating loss function based on the given parameters
    """

    def loss_fn(polygons_pred: torch.Tensor,
                polygons_gt: torch.Tensor) -> LossResult:
        """
        loss for todo
        """

        # match dim for single gate prediction
        if len(polygons_pred.shape) == 2:
            polygons_pred = polygons_pred.unsqueeze(0)

        point_losses = []
        for polygon_pred, polygon_gt in zip(polygons_pred, polygons_gt):
            polygon_gt_unpadded = remove_padded_values_from_points(polygon_gt)
            polygon_pred_unpadded = remove_padded_values_from_points(polygon_pred)

            point_losses.append(hungarian_points_to_polygon(polygon_gt_unpadded, polygon_pred_unpadded))

        point_loss = torch.stack(point_losses) * polygon_weight
        return LossResult(point_loss)

    return loss_fn
