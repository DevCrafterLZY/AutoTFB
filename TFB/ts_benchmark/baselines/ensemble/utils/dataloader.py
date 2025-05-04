import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def split_before(data: pd.DataFrame, index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
    :param index: Split index position.
    :return: Split the first and second half of the data.
    """
    return data.iloc[:index, :], data.iloc[index:, :]


def get_train_len(data_len, ratio, seq_len, horizon):
    border = int(data_len * ratio)

    # 如果数据总量足够，强制seq_len至少为10以兼容StatsForecastAutoTheta和StatsForecastAutoCES
    if data_len >= 10 + horizon:
        seq_len = max(seq_len, 10)
    # 如果数据总量不足，则保持原始的seq_len不变
    # 因为StatsForecastAutoTheta和StatsForecastAutoCES肯定没法跑了

    border = min(border, data_len - seq_len - horizon)
    return border


def train_val_split(train_data, ratio, seq_len, horizon):
    if ratio == 1:
        return train_data, None

    border = get_train_len(len(train_data), ratio, seq_len, horizon)
    train_data, valid_data = split_before(train_data, border)
    return train_data, valid_data


class EnsembleDataset(Dataset):
    def __init__(self, X, Y, Z):
        super(EnsembleDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.Z = Z

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        Z = self.Z[idx]

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        Z = torch.tensor(Z, dtype=torch.float32)

        return X, Y, Z

    def __len__(self):
        return len(self.X)


def _process_data(
        data: pd.DataFrame,
        horizon: int,
        len_before_predict: int,
        seq_len: int,
        ensemble,
        model_ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_list = []
    target_list = []

    # 如果数据总量足够（至少满足10 + horizon），强制序列长度至少为10
    # 为了保证StatsForecastAutoTheta,StatsForecastAutoCES 能正常跑因为其输入一定要大于10
    min_seq_len = len_before_predict if len(data) < 10 + horizon else max(len_before_predict, 10)

    for start in range(len(data) - min_seq_len - horizon + 1):
        seq = data.iloc[0:start + len_before_predict]
        seq_list.append(seq.astype("float32"))

        target = data.iloc[start + len_before_predict:start + len_before_predict + horizon].values
        target_list.append(target)
    inputs_list = [seq.values[-seq_len:] for seq in seq_list]
    inputs = np.stack(inputs_list, axis=0)
    model_predictions = ensemble.get_models_predicts(model_ids, seq_list, horizon)
    targets = np.stack(target_list, axis=0)

    return inputs, model_predictions, targets


def get_ensemble_dataloader(
        data: pd.DataFrame,
        horizon: int,
        len_before_predict: int,
        seq_len: int,
        ensemble,
        model_ids: List[int],
        batch_size: int,
        shuffle: bool
):
    inputs, model_predictions, targets = _process_data(data, horizon, len_before_predict, seq_len, ensemble, model_ids)
    dataset = EnsembleDataset(inputs, model_predictions, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
