import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from AutoTFB.models.model import NCF

logger = logging.getLogger(__name__)


def _rank_data(data: np.ndarray) -> np.ndarray:
    """
    Rank the input data in descending order.

    :param data: A NumPy array of data to be ranked.
    :return: A NumPy array containing the ranks of the input data, with higher values ranked higher.
    """
    data_copy = data.copy()
    ranked_data = np.argsort(np.argsort(data_copy, axis=0), axis=0)
    ranked_data[ranked_data > 30] = 30  # 为了适配lightgbm的rank
    ranked_data = max(ranked_data) - ranked_data
    return ranked_data.astype(int)


def _normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data by dividing the minimum value by each value in the dataset.

    This method ensures that each element in the dataset is normalized by its respective column's
    minimum value, making the dataset more suitable for further processing.

    :param data: The data to be normalized (typically a numpy array or pandas DataFrame).
    :return: The normalized data.
    """
    data_copy = data.copy()
    nonzero_elements = data_copy[data_copy != 0]
    if nonzero_elements.size > 0:
        epsilon = np.min(np.abs(nonzero_elements)) * 0.8
    else:
        epsilon = 1e-8
    data_copy[data_copy == 0] = epsilon
    min_values = np.min(data_copy, axis=0)
    data_copy = min_values / data_copy
    data_copy = np.nan_to_num(data_copy, nan=0.0, posinf=0.0, neginf=0.0)
    return data_copy


def preprocess_data(all_results: pd.DataFrame, target_metric: List) -> pd.DataFrame:
    """
    Preprocess the data by ranking and normalizing the target metrics.

    :param all_results: The input DataFrame containing all results.
    :param target_metric: A list of target metrics to process.
    :return: A DataFrame with normalized and ranked target metrics for each group.
    """
    all_results_copy = all_results.copy()
    all_results_copy.loc[:, target_metric] = all_results_copy.loc[:, target_metric].fillna(np.inf)

    def normalize_group(group):
        for metric in target_metric:
            group[metric + "_normalized"] = _normalize_data(group[metric].values)
        return group

    def rank_group(group):
        for metric in target_metric:
            group[metric + "_rank"] = _rank_data(group[metric].values)
        return group

    all_results_copy = all_results_copy.groupby("file_name", group_keys=False).apply(rank_group)
    all_results_copy = all_results_copy.groupby("file_name", group_keys=False).apply(normalize_group)
    return all_results_copy


def get_kfold_train_test_data(
        all_results: pd.DataFrame,
        k_folds: int = 5
) -> List:
    """
    Split the dataset into k-folds for cross-validation.

    :param all_results: A DataFrame containing all results, with each row representing a dataset entry.
    :param k_folds: The number of folds to divide the data into (default is 5).
    :return: A tuple containing two lists:
             - fold_data: A list of tuples, each containing the training and test data for a fold.
             - test_indices: A list of indices of the test data across all folds.
    """
    dataset_names = all_results["file_name"].unique().tolist()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    dataset_name_to_indices = {}
    for dataset_name in dataset_names:
        dataset_name_to_indices[dataset_name] = all_results[all_results["file_name"] == dataset_name].index.tolist()

    dataset_name_splits = list(kf.split(dataset_names))

    fold_data = []

    for train_dataset_names_idx, test_dataset_names_idx in dataset_name_splits:

        train_dataset_names = [dataset_names[i] for i in train_dataset_names_idx]
        test_dataset_names = [dataset_names[i] for i in test_dataset_names_idx]

        train_results = all_results[all_results["file_name"].isin(train_dataset_names)]
        test_results = all_results[all_results["file_name"].isin(test_dataset_names)]

        fold_data.append((train_results, test_results))
    return fold_data


def _get_NCF_data(
        dataset_feature: Dict,
        model_names_to_index: Dict,
        all_results: pd.DataFrame,
        target_metric: List,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Extract time features, model IDs, and normalized target metrics from the dataset.

    :param dataset_feature: A dictionary containing the features of each dataset.
    :param model_names_to_index: A dictionary mapping model names to model indices.
    :param all_results: A DataFrame containing all results, including the target metrics.
    :param target_metric: A list of target metrics to be normalized.
    :return: A tuple of three NumPy arrays:
             - The time features for each dataset.
             - The model IDs corresponding to the models used in the dataset.
             - The normalized target metrics for each dataset.
    """
    time_features = []
    model_ids = []
    normalized_target_metrics = []

    normalized_target_metric = [metric + "_normalized" for metric in target_metric]

    for index, row in tqdm(all_results.iterrows(), total=len(all_results)):
        dataset_name = row["file_name"]
        model_name = row["model_name"]

        time_features.append(dataset_feature[dataset_name].flatten())
        model_ids.append(model_names_to_index[model_name])
        normalized_target_metrics.append(row[normalized_target_metric].values.astype(float))

    return np.vstack(time_features), np.vstack(model_ids).squeeze(), np.vstack(normalized_target_metrics)


def get_NCF_dataloader(
        dataset_feature: Dict,
        model_names_to_index: Dict,
        all_results: pd.DataFrame,
        target_metric: List,
        batch_size: int = 32,
        val_ratio = 0.1,
) -> (DataLoader, DataLoader):
    """
    Prepare DataLoader objects for training and validation.

    :param dataset_feature: A dictionary containing the extracted features of each dataset.
    :param model_names_to_index: A dictionary mapping model names to unique indices.
    :param all_results: A DataFrame containing all results, including target metrics.
    :param target_metric: A list of target metrics to be used for training.
    :param batch_size: The batch size for training and validation.
    :return: A tuple containing:
             - train_dataloader: DataLoader for training data.
             - val_dataloader: DataLoader for validation data.
    """
    time_features, model_ids, normalized_target_metrics = _get_NCF_data(
        dataset_feature,
        model_names_to_index,
        all_results,
        target_metric
    )
    time_features_tensor = torch.tensor(time_features, dtype=torch.float32)
    model_ids_tensor = torch.tensor(model_ids, dtype=torch.int64)
    y_tensor = torch.tensor(normalized_target_metrics, dtype=torch.float32)

    x_train, x_val, model_ids_train, model_ids_val, y_train, y_val = train_test_split(
        time_features_tensor, model_ids_tensor, y_tensor,
        test_size=val_ratio, random_state=42
    )

    train_dataset = TensorDataset(x_train, model_ids_train, y_train)
    val_dataset = TensorDataset(x_val, model_ids_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def _get_lightgbm_data(
        dataset_feature: Dict,
        all_results: pd.DataFrame,
        ncf: NCF,
        target_metric: List,
        names_to_index: Dict,
        batch_size: int = 512,
) -> (np.ndarray, np.ndarray):
    """
    Prepare features and target metrics for LightGBM by processing data in batches.

    :param dataset_feature: A dictionary containing the features of each dataset.
    :param all_results: A DataFrame containing all results, including target metrics.
    :param ncf: The NCF model used for generating embeddings for time features and model IDs.
    :param target_metric: A list of target metrics for ranking.
    :param batch_size: The batch size to process the data in.
    :return: A tuple of two NumPy arrays:
             - x: The features (concatenation of time series and model embeddings).
             - rank_y: The target metrics (ranked version of the target metrics).
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = []
    rank_y = []

    time_features_batch = []
    model_indexes_batch = []

    rank_target_metric = [metric + "_rank" for metric in target_metric]

    for index, row in tqdm(all_results.iterrows(), total=len(all_results)):
        dataset_name = row["file_name"]
        model_name = row["model_name"]
        model_index = names_to_index[model_name]

        # Collect the features in batches
        time_features_batch.append(torch.tensor(dataset_feature[dataset_name]).flatten())
        model_indexes_batch.append(model_index)

        # Once we reach the batch size, process the batch
        if len(time_features_batch) >= batch_size:
            time_features_tensor = torch.stack(time_features_batch).to(device)
            model_indexes_tensor = torch.tensor(model_indexes_batch).to(device)

            # Process time features and model embeddings in batches
            tsvec_batch = ncf.tsvec_encoder.to(device)(time_features_tensor).detach().cpu().numpy()
            model_emb_batch = ncf.model_embedding.to(device)(model_indexes_tensor).detach().cpu().numpy()

            # Concatenate and store results
            for tsvec, model_emb, row in zip(tsvec_batch, model_emb_batch,
                                             all_results.iloc[len(x):len(x) + batch_size].iterrows()):
                conbine_emb = np.concatenate((tsvec, model_emb))
                rank_metrics = row[1][rank_target_metric].values

                x.append(conbine_emb)
                rank_y.append(rank_metrics)

            # Reset the batches
            time_features_batch = []
            model_indexes_batch = []

    # Process any remaining data that didn't fit into the last batch
    if time_features_batch:
        time_features_tensor = torch.stack(time_features_batch).to(device)
        model_indexes_tensor = torch.tensor(model_indexes_batch).to(device)

        tsvec_batch = ncf.tsvec_encoder.to(device)(time_features_tensor).detach().cpu().numpy()
        model_emb_batch = ncf.model_embedding.to(device)(model_indexes_tensor).detach().cpu().numpy()

        for tsvec, model_emb, row in zip(tsvec_batch, model_emb_batch, all_results.iloc[len(x):].iterrows()):
            conbine_emb = np.concatenate((tsvec, model_emb))
            rank_metrics = row[1][rank_target_metric].values

            x.append(conbine_emb)
            rank_y.append(rank_metrics)

    return np.vstack(x), np.vstack(rank_y)


def get_lightgbm_data(
        ncf: NCF,
        dataset_feature: Dict,
        train_results: pd.DataFrame,
        test_results: pd.DataFrame,
        target_metric: List,
        names_to_index: Dict,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Prepare the training and testing data for LightGBM by calling the helper function
    `_get_lightgbm_data` for both train and test datasets.

    :param ncf: The NCF model used to generate embeddings for time series and model IDs.
    :param dataset_feature: A dictionary containing the features of each dataset.
    :param train_results: A DataFrame containing the training results, including the target metrics.
    :param test_results: A DataFrame containing the testing results, including the target metrics.
    :param target_metric: A list of target metrics for ranking.
    :param names_to_index: A dictionary mapping model names to unique indices.
    :return: A tuple containing four NumPy arrays:
             - train_x: The features (concatenation of time series and model embeddings) for training.
             - train_y: The target metrics (ranked) for training.
             - test_x: The features (concatenation of time series and model embeddings) for testing.
             - test_y: The target metrics (ranked) for testing.
    """
    train_x, train_y = _get_lightgbm_data(
        dataset_feature=dataset_feature,
        all_results=train_results,
        target_metric=target_metric,
        ncf=ncf,
        names_to_index=names_to_index,
    )
    test_x, test_y = _get_lightgbm_data(
        dataset_feature=dataset_feature,
        all_results=test_results,
        target_metric=target_metric,
        ncf=ncf,
        names_to_index=names_to_index,
    )

    return train_x, train_y, test_x, test_y
