import pandas as pd
import numpy as np
from numba import jit, njit
import numba


def points_in_polygon(events: pd.DataFrame, polygon_points: np.ndarray, x_idx: int, y_idx: int, scale : bool = True) -> np.ndarray:
    """
    computes if the given events are inside the given polygon based on the given x and y axis
    """
    # events must be scaled to be on same range with polygons
    event_x_y = np.array(events.iloc[:, [x_idx, y_idx]])

    if scale:
        event_x_y /= 4.5

    # inside_sm needs polygons to be closed --> last points == first point --> add first point at last element
    polygon_points = np.concatenate([polygon_points, [polygon_points[0]]])

    mask = is_inside_sm_parallel(event_x_y, polygon_points)

    return mask

# region [code taken from https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py and https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python]
# I found this code on the internet, it computes points in a polygon super fast!
# return value:
# 0 - the point is outside the polygon
# 1 - the point is inside the polygon 
# 2 - the point is one edge (boundary)

@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 1

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (point[0] == polygon[jj][0] or (dy == 0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0]) <= 0)):
                return 1

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return intersections & 1


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean)
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon, points[i])
    return D

# endregion