import torch
import torch.nn.functional as f

def point_to_polygon_edges(point: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    calculates the distance between a point and all edges of a polygon
    """
    vertices_start = polygon
    vertices_end = polygon.clone().roll(1, dims=-2)

    d_start_to_point = torch.linalg.norm(vertices_start - point, dim=-1)
    d_end_to_point = torch.linalg.norm(vertices_end - point, dim=-1)
    d_start_to_end = torch.linalg.norm(vertices_start - vertices_end, dim=-1)

    # distance = torch.pow(d_start_to_point + d_end_to_point - d_start_to_end, 2)
    distance = torch.abs(d_start_to_point) + torch.abs(d_end_to_point) - torch.abs(d_start_to_end)

    return distance


def point_to_polygon_edges_perpendicular(point: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    caclulates the perpendicular distance between a point and all edges of a polygon
    """

    # point = [5,3]
    # polygon = [[1,2], [2,4], [3,4], [3,1]]
    # polygon = [[2,4], [3,4], [3,1], [1,2]]
    # B = [[1,2], ]
    vertices_start = polygon    
    # roll vertices to get pairwise edges
    vertices_end = polygon.clone().roll(1, dims=-2)

    vertices = vertices_start - vertices_end
    diff_point_vertices_end = point - vertices_end
    vertices_normed = torch.norm(vertices, dim=-1)
    vertices_padded = f.pad(vertices, (0, 1))
    diff_point_vertices_end_padded = f.pad(diff_point_vertices_end, (0, 1))
    cross_result = torch.cross(vertices_padded, diff_point_vertices_end_padded, dim=-1)
    norm_cross = torch.norm(cross_result, dim=-1)
    distances = norm_cross / vertices_normed

    return distances


def point_to_polygon_distance(point: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    returns the minimum perpendicular distance between a point and all edges of a polygon
    """

    distances = point_to_polygon_edges(point, polygon)
    distances = distances.nan_to_num(float("inf"))

    if len(distances.shape) < 2:
        distances = distances.unsqueeze(dim=-1)

    # min_per_last_dim = torch.min(distances, dim=1)[0]
    min_per_last_dim = torch.min(distances)

    return min_per_last_dim


def points_to_polygon(points: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """
    calculates the sumed minimal perpendicular distance from all points to all edges of a given polygon
    """
    # this method is not parallelized because of variable polygon size in GT --> maybe use mask in future
    res_list = []
    for i, point in enumerate(points):
        if point[0] != -1:  # only if point is predicted
            res_list.append(point_to_polygon_distance(point, polygon))

    return torch.Tensor.sum(torch.stack(res_list))
