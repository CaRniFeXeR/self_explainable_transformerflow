import numpy as np


def interpolate_polygon(polygon: np.ndarray, n_needed_points: int) -> np.ndarray:
    """
    interpolates a given polygon to result in a polygon with n_needed_points, such that the number of interpolated points are
    equally spread over all edges.
    e.g. if 4 points are given but 9 are needed the first edge is interpolated with 2 points and all other edges with one.
    """

    n_original_points = len(polygon)
    n_points_to_interpolate = n_needed_points - n_original_points

    if n_points_to_interpolate == 0:
        return polygon

    if n_points_to_interpolate < 0:
        raise ValueError(f"more points given in the polygon than expected: given {len(polygon)} expected: {n_needed_points}")

    minimum_interpolation_per_edge = n_points_to_interpolate // n_original_points
    n_edges_with_extra_interpolation = n_points_to_interpolate % n_original_points

    extended_points = []

    for i in range(0, n_original_points):
        point_idx_s = i
        point_idx_e = (i+1) % n_original_points

        extended_points.append(polygon[i])

        # each edge has at least this amount of interpolations
        n_interpolations = minimum_interpolation_per_edge
        if i < n_edges_with_extra_interpolation:
            # there are some edges that have one interpolation point extra
            n_interpolations += 1

        for j in range(1, n_interpolations + 1):
            interpolated_point = polygon[point_idx_s] + ((polygon[point_idx_e] - polygon[point_idx_s]) * (j / (n_interpolations + 1)))

            extended_points.append(interpolated_point)

    return np.array(extended_points)


def reduce_polygon_length(polygon: np.ndarray, n_needed_points: int) -> np.ndarray:
    """
    reduces a given polygon to n_needed_points by equally omitting points.
    """

    n_current = len(polygon)
    n_to_omitt = n_current - n_needed_points

    if n_to_omitt == 0: 
        # nothing to reduce
        return polygon

    if n_to_omitt < 0:
        raise ValueError(f"less points given than needed! given: '{n_current}' n_needed_points: '{n_needed_points}'")

    if n_to_omitt <= n_current / 2:
        if n_current % n_to_omitt > 0:
            raise ValueError(f"polygon reduction cannot be equally distributed. '{n_current}' '{n_to_omitt}'")

        positions_to_omitt = n_current / n_to_omitt

        new_polygon = []

        for i in range(n_current):
            if (i + 1) % positions_to_omitt > 0:
                new_polygon.append(polygon[i])
        
        return np.array(new_polygon)
    else:
        if n_current % n_needed_points > 0:
            raise ValueError(f"polygon reduction cannot be equally distributed. '{n_current}' '{n_needed_points}'")

        positions_to_keep = n_current / n_needed_points

        new_polygon = []

        for i in range(n_current):
            if (i + 1) % positions_to_keep == 0:
                new_polygon.append(polygon[i])
        
        return np.array(new_polygon)