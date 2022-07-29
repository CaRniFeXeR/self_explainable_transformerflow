import numpy as np

"""
holds formulas for various data transformations used in the pipline
"""


def normalise_events(events, normalize_mean, normalize_sd):
    """
    normalises given events to mean = 0 and sd = 1 based on given mean and sd.
    """

    means = np.array(normalize_mean)
    sds = np.array(normalize_sd)
    return (events - means) / sds


def denormalise_events(events, normalize_mean, normalize_sd):
    """
    denormalises events from mean = 0 and sd =1 to the original values based on given mean and sd.
    """

    means = np.array(normalize_mean)
    sds = np.array(normalize_sd)
    return events * sds + means


def scale_polygon_points(polygon: np.ndarray, min: float, max: float):
    """
    executes min-max-scaling for given polygon points min and max value
    """

    return (polygon - min) / (max - min)


def unscale_polygon_points(polygon: np.ndarray, min: float, max: float):
    """
    untransforms polygon points from min-max-scaling based on given min and max value
    """

    return polygon * (max - min) + min
