import torch


def remove_padded_values_from_points(points: torch.Tensor) -> torch.Tensor:
    result = []
    for point in points:
        if point[0] != -1:
            result.append(point)

    return torch.stack(result)
