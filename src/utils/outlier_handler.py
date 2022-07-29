import pandas as pd
import numpy as np
from scipy.stats import chi2
from ..datastructures.configs.outlierhandlerconfig import OutlierHandlerConfig
from scipy.linalg import solve_triangular


class OutlierHandler:

    def __init__(self, config: OutlierHandlerConfig) -> None:
        self.config = config

    def compute_mahalanobis(self, data: np.ndarray, cov: np.ndarray = None):

        mu = np.mean(data, axis=0)
        if not cov:
            cov = np.cov(data.T)
        L = np.linalg.cholesky(cov)
        d = data - mu
        z = solve_triangular(L, d.T, lower=True, check_finite=False, overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def test_outlier(self, mahalanobis_distance: np.ndarray, degrees_of_freedom: int, alpha: float = 0.05):
        return (1 - chi2.cdf(mahalanobis_distance, degrees_of_freedom)) < alpha

    def get_non_outliers(self, events: pd.DataFrame):

        if len(events) < self.config.n_events_threshold:
            return np.ones(len(events))

        maha_d = self.compute_mahalanobis(np.array(events))
        outlier_mask = self.test_outlier(maha_d, degrees_of_freedom=events.shape[1],  alpha=self.config.alpha)

        return outlier_mask == False
