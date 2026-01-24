import torch
import numpy as np
from skorch.dataset import ValidSplit
from sklearn.model_selection import train_test_split
from EarlyStop import CustomEarlyStopCallback    
from catboost import CatBoostRegressor


class Probability():
    def __init__(self, Q, model_size, quantiles, model_type, device):
        self.Q = Q
        self.model_size = model_size
        self.quantiles = quantiles
        self.model_type = model_type
        self.device = device

    def quantile_regression(self, used_data, used_label, seed, x_test, X):
        key = "MultiQuantile:alpha=" + ",".join([str(q) for q in self.quantiles])
        if self.model_type == "Tree":
            model = CatBoostRegressor(iterations=self.model_size,
                                    quantiles=self.quantiles, 
                                    loss_function=key,
                                    random_state=seed,
                                    verbose=0)

        else:
            assert self.model_type == "Tree", 'The model_type keywords must be "Tree" type.'

        model.fit(used_data, used_label, callbacks=[CustomEarlyStopCallback(key)])

        preds = model.predict(x_test)

        return model, preds

    def get_interval_ranges(self, k, range_min_value, range_max_value):
        assert range_max_value >= range_min_value, 'The maximum values of predictions must be larger or equal than the minimun values of the predictions.'
        ranges = np.linspace(range_min_value, range_max_value, k+1)

        return ranges

    def get_predictive_distribution(self, unlabeled_prediction, Q, K, ranges, range_max_value):
        interval_values = np.zeros(shape=(K), dtype=int)
        for j in np.arange(unlabeled_prediction.shape[0]):
            # Each intervals:
            for i in np.arange(0, K, 1):
                start = ranges[i]
                end = ranges[i+1]
                if start <= unlabeled_prediction[j] < end:
                    interval_values[i] += 1
                if unlabeled_prediction[j] == range_max_value and i+1 == K:
                    interval_values[i] += 1
                    
        assert np.sum(interval_values) == Q-1, 'Warning: The interval split has errors.'

        return interval_values/(Q-1)