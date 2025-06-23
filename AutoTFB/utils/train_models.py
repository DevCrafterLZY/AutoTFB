import os
from typing import Dict, List

import lightgbm
import pandas as pd

from AutoTFB.NCF import NCF_model
from AutoTFB.models.model import NCF
from common.constant import LIGHTGBM_MODEL_CHECKPOINT_PATH
from utils import get_NCF_dataloader, get_lightgbm_data


def train_NCF(
        config,
        dataset_feature: Dict,
        model_names_to_index: Dict,
        train: pd.DataFrame,
        target_metric: List,
) -> NCF:
    """
    Train the NCF model using the provided training data.

    :param config: The configuration object containing model hyperparameters and settings.
    :param dataset_feature: A dictionary containing the dataset features.
    :param model_names_to_index: A dictionary mapping model names to indices.
    :param train: The training data.
    :param target_metric: The target metrics to predict.
    :return: The trained NCF model.
    """
    train_dataloader, val_dataloader = get_NCF_dataloader(
        dataset_feature=dataset_feature,
        model_names_to_index=model_names_to_index,
        all_results=train,
        target_metric=target_metric,
        batch_size=config.batch_size,
    )
    model = NCF_model(config)
    model.train(train_dataloader, val_dataloader)
    return model.model


def test_NCF(
        config,
        dataset_feature: Dict,
        model_names_to_index: Dict,
        train: pd.DataFrame,
        target_metric: List,
) -> NCF:
    """
    Train the NCF model using the provided training data.

    :param config: The configuration object containing model hyperparameters and settings.
    :param dataset_feature: A dictionary containing the dataset features.
    :param model_names_to_index: A dictionary mapping model names to indices.
    :param train: The training data.
    :param target_metric: The target metrics to predict.
    :return: The trained NCF model.
    """
    train_dataloader, val_dataloader = get_NCF_dataloader(
        dataset_feature=dataset_feature,
        model_names_to_index=model_names_to_index,
        all_results=train,
        target_metric=target_metric,
        batch_size=config.batch_size,
    )
    model = NCF_model(config)
    model.train(train_dataloader, val_dataloader)
    return model.model


def train_test_gbm(
        ncf: NCF,
        params: Dict,
        dataset_feature: Dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_metrics: List,
        model_names_to_index: Dict,
        gbm_params_path: str,
):
    """
    Train and test the LightGBM model using the provided dataset and parameters.

    :param ncf: The pre-trained NCF model used for feature extraction.
    :param params: The hyperparameters for the LightGBM model.
    :param dataset_feature: A dictionary containing the dataset features.
    :param train: The training data.
    :param test: The testing data.
    :param target_metrics: The target metrics to optimize during training.
    :param model_names_to_index: A dictionary mapping model names to indices.
    :return: The predictions from the trained LightGBM model on the test dataset.
    """
    ncf.eval()
    train_x, train_y, test_x, test_y = get_lightgbm_data(
        ncf=ncf,
        dataset_feature=dataset_feature,
        train_results=train,
        test_results=test,
        target_metric=target_metrics,
        names_to_index=model_names_to_index,
    )

    train_groups = train.groupby('file_name').size().tolist()

    train_dataset = lightgbm.Dataset(train_x, label=train_y.squeeze(), group=train_groups)
    gbm = lightgbm.train(
        params, train_dataset, num_boost_round=100, valid_sets=[train_dataset],
        callbacks=[lightgbm.log_evaluation, lightgbm.early_stopping(10, first_metric_only=False)]
    )

    gbm_folder_path = os.path.dirname(gbm_params_path)
    if not os.path.exists(gbm_folder_path):
        os.makedirs(gbm_folder_path, exist_ok=True)
    gbm.save_model(gbm_params_path)
    pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    return pred
