MODEL_CATEGORY = {
    "XGBModel": "darts",
    "Triformer": "time_series_library",
    "TimesNet": "time_series_library",
    "TiDEModel": "darts",
    "TCNModel": "darts",
    "StatsForecastAutoTheta": "darts",
    "StatsForecastAutoETS": "darts",
    "StatsForecastAutoCES": "darts",
    "RNNModel": "darts",
    "RandomForest": "darts",
    "PatchTST": "time_series_library",
    "NLinear": "time_series_library",
    "NHiTSModel": "darts",
    "NBEATSModel": "darts",
    "NaiveSeasonal": "darts",
    "NaiveMovingAverage": "darts",
    "NaiveMean": "darts",
    "NaiveDrift": "darts",
    "LinearRegressionModel": "darts",
    "Linear": "time_series_library",
    "KalmanForecaster": "darts",
    "Informer": "time_series_library",
    "FiLM": "time_series_library",
    "FEDformer": "time_series_library",
    "DLinear": "time_series_library",
    "Crossformer": "time_series_library",
    "BlockRNNModel": "darts",
    "AutoARIMA": "darts",
    "Nonstationary_Transformer": "time_series_library",
    "LightGBMModel": "darts",
}

CATEGORY_ADAPTER = {
    "time_series_library": "transformer_adapter",
}

LEN2CONFIG = {
    6: "fixed_forecast_config_yearly.json",
    8: "fixed_forecast_config_other.json",
    13: "fixed_forecast_config_weekly.json",
    14: "fixed_forecast_config_daily.json",
    18: "fixed_forecast_config_monthly.json",
    48: "fixed_forecast_config_hourly.json",
}

HORIZON2SEQ_LEN = {
    6: 7,
    8: 10,
    13: 16,
    14: 17,
    18: 22,
    48: 60,
}

MODEL_TYPE = {
    "PatchTST": "time_series_library",
    "NLinear": "time_series_library",
    "Triformer": "time_series_library",
    "TimesNet": "time_series_library",
    "Informer": "time_series_library",
    "FiLM": "time_series_library",
    "FEDformer": "time_series_library",
    "DLinear": "time_series_library",
    "Crossformer": "time_series_library",
    "Linear": "time_series_library",
    "Nonstationary_Transformer": "time_series_library",

    "NHiTSModel": "darts_deep",
    "NBEATSModel": "darts_deep",
    "RNNModel": "darts_deep",
    "BlockRNNModel": "darts_deep",
    "TiDEModel": "darts_deep",
    "TCNModel": "darts_deep",

    "RandomForest": "darts_regression",
    "XGBModel": "darts_regression",
    "LightGBMModel": "darts_regression",
    "LinearRegressionModel": "darts_regression",

    "StatsForecastAutoTheta": "darts_stat",
    "StatsForecastAutoETS": "darts_stat",
    "StatsForecastAutoCES": "darts_stat",
    "NaiveSeasonal": "darts_stat",
    "NaiveMovingAverage": "darts_stat",
    "NaiveMean": "darts_stat",
    "NaiveDrift": "darts_stat",
    "KalmanForecaster": "darts_stat",
    "AutoARIMA": "darts_stat",
}


def darts_deep_default_config(horizon, horizon2seq_len):
    return {
        "input_chunk_length": horizon2seq_len[horizon],
        "output_chunk_length": horizon,
        "norm": 0,
        "pl_trainer_kwargs": {
            "accelerator": "gpu", "devices": -1
        }
    }


def darts_regression_default_config(horizon, horizon2seq_len):
    return {
        "lags": horizon2seq_len[horizon],
        "output_chunk_length": 1,
        "norm": 0,
    }


def darts_stat_default_config(horizon, horizon2seq_len):
    return {
        "norm": 0,
    }


def tslib_default_config(horizon, horizon2seq_len):
    return {
        "horizon": horizon,
        "norm": 0,
    }


TYPE_DEFAULT_CONFIG = {
    "darts_deep": darts_deep_default_config,
    "darts_regression": darts_regression_default_config,
    "darts_stat": darts_stat_default_config,
    "time_series_library": tslib_default_config,
}

DEEP_LEARNING = "deep_learning"
NO_DEEP_LEARNING = "no_deep_learning"
STATISTICAL_LEARNING = "statistical_learning"
MACHINE_LEARNING = "machine_learning"

# MODEL_LEARNING_TYPE = {
#     "Nonstationary_Transformer": DEEP_LEARNING,
#     "DLinear": DEEP_LEARNING,
#     "TCNModel": DEEP_LEARNING,
#     "TimesNet": DEEP_LEARNING,
#     "NLinear": DEEP_LEARNING,
#     "Triformer": DEEP_LEARNING,
#     "NHiTSModel": DEEP_LEARNING,
#     "Crossformer": DEEP_LEARNING,
#     "PatchTST": DEEP_LEARNING,
#     "NBEATSModel": DEEP_LEARNING,
#     "Informer": DEEP_LEARNING,
#     "FEDformer": DEEP_LEARNING,
#     "FiLM": DEEP_LEARNING,
#     "BlockRNNModel": DEEP_LEARNING,
#     "RNNModel": DEEP_LEARNING,
#     "TiDEModel": DEEP_LEARNING,
#
#     "RandomForest": MACHINE_LEARNING,
#     "LinearRegressionModel": MACHINE_LEARNING,
#     "Linear": MACHINE_LEARNING,
#     "XGBModel": MACHINE_LEARNING,
#     "LightGBMModel": MACHINE_LEARNING,
#
#     "KalmanForecaster": STATISTICAL_LEARNING,
#     "NaiveMovingAverage": STATISTICAL_LEARNING,
#     "NaiveMean": STATISTICAL_LEARNING,
#     "NaiveSeasonal": STATISTICAL_LEARNING,
#     "NaiveDrift": STATISTICAL_LEARNING,
#     "StatsForecastAutoCES": STATISTICAL_LEARNING,
#     "StatsForecastAutoTheta": STATISTICAL_LEARNING,
#     "AutoARIMA": STATISTICAL_LEARNING,
#     "StatsForecastAutoETS": STATISTICAL_LEARNING,
#
# }


# MODEL_LEARNING_TYPE = {
#     "Nonstationary_Transformer": DEEP_LEARNING,
#     "DLinear": DEEP_LEARNING,
#     "TCNModel": DEEP_LEARNING,
#     "TimesNet": DEEP_LEARNING,
#     "NLinear": DEEP_LEARNING,
#     "Triformer": DEEP_LEARNING,
#     "NHiTSModel": DEEP_LEARNING,
#     "Crossformer": DEEP_LEARNING,
#     "PatchTST": DEEP_LEARNING,
#     "NBEATSModel": DEEP_LEARNING,
#     "Informer": DEEP_LEARNING,
#     "FEDformer": DEEP_LEARNING,
#     "FiLM": DEEP_LEARNING,
#     "BlockRNNModel": DEEP_LEARNING,
#     "RNNModel": DEEP_LEARNING,
#     "TiDEModel": DEEP_LEARNING,
#
#     "RandomForest": MACHINE_LEARNING,
#     "LinearRegressionModel": MACHINE_LEARNING,
#     "Linear": MACHINE_LEARNING,
#     "XGBModel": MACHINE_LEARNING,
#     "LightGBMModel": MACHINE_LEARNING,
#
#     "KalmanForecaster": STATISTICAL_LEARNING,
#     "NaiveMovingAverage": STATISTICAL_LEARNING,
#     "NaiveMean": STATISTICAL_LEARNING,
#     "NaiveSeasonal": STATISTICAL_LEARNING,
#     "NaiveDrift": STATISTICAL_LEARNING,
#     "StatsForecastAutoCES": STATISTICAL_LEARNING,
#     "StatsForecastAutoTheta": STATISTICAL_LEARNING,
#     "AutoARIMA": STATISTICAL_LEARNING,
#     "StatsForecastAutoETS": STATISTICAL_LEARNING,
#
# }


MODEL_LEARNING_TYPE = {
    "PatchTST": DEEP_LEARNING,
    "NLinear": DEEP_LEARNING,
    "Triformer": DEEP_LEARNING,
    "TimesNet": DEEP_LEARNING,
    "Informer": DEEP_LEARNING,
    "FiLM": DEEP_LEARNING,
    "FEDformer": DEEP_LEARNING,
    "DLinear": DEEP_LEARNING,
    "Crossformer": DEEP_LEARNING,
    "Linear": DEEP_LEARNING,
    "Nonstationary_Transformer": DEEP_LEARNING,

    "NHiTSModel": DEEP_LEARNING,
    "NBEATSModel": DEEP_LEARNING,
    "RNNModel": DEEP_LEARNING,
    "BlockRNNModel": DEEP_LEARNING,
    "TiDEModel": DEEP_LEARNING,
    "TCNModel": DEEP_LEARNING,

    "RandomForest": DEEP_LEARNING,
    "XGBModel": DEEP_LEARNING,
    "LightGBMModel": DEEP_LEARNING,
    "LinearRegressionModel": DEEP_LEARNING,

    "StatsForecastAutoTheta": NO_DEEP_LEARNING,
    "StatsForecastAutoETS": NO_DEEP_LEARNING,
    "StatsForecastAutoCES": NO_DEEP_LEARNING,
    "NaiveSeasonal": NO_DEEP_LEARNING,
    "NaiveMovingAverage": NO_DEEP_LEARNING,
    "NaiveMean": NO_DEEP_LEARNING,
    "NaiveDrift": NO_DEEP_LEARNING,
    "KalmanForecaster": NO_DEEP_LEARNING,
    "AutoARIMA": NO_DEEP_LEARNING,

}
