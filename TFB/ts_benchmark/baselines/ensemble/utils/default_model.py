import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class DefaultModel:
    def __init__(self):
        pass

    @property
    def model_name(self):
        return "DefaultModel"

    @staticmethod
    def forecast(horizon: int, train: pd.DataFrame) -> np.ndarray:
        """
        Use Simple Exponential Smoothing to forecast the next `horizon` values of the time series.
        If there are fewer than 2 data points, repeat the last value.

        :param horizon: Forecast horizon (int)
        :param train: Training data (pd.DataFrame), assumed to be a single-column time series
        :return: Forecast results (np.ndarray)
        """

        series = train.iloc[:, 0].to_numpy(dtype=float)  # Convert to float numpy array

        if series.size == 0:
            raise ValueError("The training data is empty, and thus prediction is not possible.")

        if len(series) < 2:
            return np.full((horizon, 1), series[-1])

        model = SimpleExpSmoothing(series).fit()
        return model.forecast(horizon).reshape(-1, 1)
