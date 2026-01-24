import numpy as np


class Evaluation():
    def __init__(self, quantiles, K, Q):
        self.quantiles = quantiles
        self.K = K
        self.Q = Q

    def evaluation_metrics(self, prediction_, X_test, y_test, get_predictive_distribution, get_interval_ranges):

        # Point prediction of the median value
        # The index of the median values, the quantiles is 0.01-0.99, median=0.5, index=49
        q_idx = self.quantiles.index(0.5)
        # The prediction point of the 50 quantile:
        prediction_point = prediction_[:, q_idx]
            
        return prediction_point
        