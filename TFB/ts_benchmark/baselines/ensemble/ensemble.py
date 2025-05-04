import concurrent.futures
import logging
import threading
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ts_benchmark.baselines.ensemble.models.ensemble_model import EnsembleModel, msmape_loss
from ts_benchmark.baselines.ensemble.utils.dataloader import (
    get_ensemble_dataloader, get_train_len
)
from ts_benchmark.baselines.ensemble.utils.default_model import DefaultModel
from ts_benchmark.baselines.ensemble.utils.ensemble_constant import ENSEMBLE_MODELS
from ts_benchmark.baselines.ensemble.utils.get_ensemble_models import get_ensemble_models
from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.models import ModelBase, ModelFactory
from ts_benchmark.models.model_base import BatchMaker

CRITERION = {
    "msmape": msmape_loss,
    "mae": nn.L1Loss(),
    "mse": nn.MSELoss(),
}

DEFAULT_ENSEMBLE_BASED_HYPER_PARAMS = {
    "batch_size": 32,
    "num_workers": 0,
    "strategy": "weighted",
    "models": [],
    "norm": False,
    "num_epochs": 100,
    "lr": 0.01,
    "patience": 10,
    "lradj": "type1",
    "ensemble_train_ratio_in_tv": 0.8,
    "min_val_samples_num": 10,
    "criterion": "msmape",
    "max_workers": None,
    "select_best_model": True,
}
logger = logging.getLogger(__name__)


class EnsembleConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_ENSEMBLE_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


def _train_model(model_factory, train, train_ratio_in_tv, horizon):
    model = model_factory()
    if hasattr(model, "forecast_fit"):
        model.forecast_fit(train, train_ratio_in_tv)
    else:
        model.fit(train, train_ratio_in_tv)

    temp = model.forecast(horizon, train)
    if np.any(np.isnan(temp)):
        raise ValueError("NaN values found in model parameters!")

    return model


def _get_model_predicts(model, predict_len, inputs):
    if model.batch_forecast.__annotations__.get("not_implemented_batch"):
        return _get_model_single_predicts(model, predict_len, inputs)
    else:
        return _get_model_batch_predicts(model, predict_len, inputs)


class _BatchMaker(BatchMaker):

    def __init__(self, series: List[pd.DataFrame]):
        self.series = series
        self.current_sample_count = 0

    def make_batch(self, batch_size: int, win_size: int) -> dict:
        """
        Return a batch of data with index and column to be used for batch prediction.

        :param batch_size: The size of batch.
        :param win_size: The length of data used for prediction.
        :return: a batch of data and its time stamps.
        """

        all_data = self.series[self.current_sample_count: self.current_sample_count + batch_size]
        input_data = [data.values[-win_size:] for data in all_data]
        input_data = np.stack(input_data, axis=0)
        time_stamps = [data.index[-win_size:] for data in all_data]
        time_stamps = np.stack(time_stamps, axis=0)
        self.current_sample_count += batch_size
        return {"input": input_data, "time_stamps": time_stamps}

    def has_more_batches(self) -> bool:
        """
        Check if there are more batches to process.

        :return: True if there are more batches, False otherwise.
        """
        return self.current_sample_count < len(self.series)


def _get_model_batch_predicts(model, predict_len, inputs):
    all_predicts = []
    batch_maker = _BatchMaker(inputs)
    while batch_maker.has_more_batches():
        predicts = model.batch_forecast(predict_len, batch_maker)
        all_predicts.append(predicts)
    all_predicts = np.concatenate(all_predicts, axis=0)
    return all_predicts


def _get_model_single_predicts(model, predict_len, inputs):
    predicts = []
    for input in tqdm(inputs, position=0, leave=True):
        try:
            predict = model.forecast(predict_len, input)
            if not isinstance(predict, np.ndarray) or not np.isfinite(predict).all():
                logger.error(f"Model {getattr(model, 'model_name', type(model).__name__)} produced invalid prediction")
                return None
            predicts.append(predict)
        except Exception as e:
            logger.error(f"Model {getattr(model, 'model_name', type(model).__name__)} prediction failed: {str(e)}")
            return None
    return np.stack(predicts, axis=0)


def _get_val_samples_num(data_len, ratio, seq_len, horizon):
    len_before_val = get_train_len(data_len, ratio, seq_len, horizon)
    val_samples_num = data_len - len_before_val - seq_len - horizon + 1
    return val_samples_num


class Ensemble(ModelBase):
    def __init__(self, **kwargs):
        self.config = EnsembleConfig(**kwargs)
        assert self.config.strategy in ["weighted", "average"]
        assert 0 <= self.config.ensemble_train_ratio_in_tv < 1
        self.criterion = CRITERION.get(self.config.criterion, msmape_loss)
        self.best_model = None
        self.trained_models = []
        self.available_model_ids = []
        self.ensemble_model_ids = []
        self.early_stopping = EarlyStopping(patience=self.config.patience)
        self.ensemble_model: EnsembleModel

    @property
    def model_name(self):
        return "Ensemble"

    def forecast_fit(
            self, train_val_data: pd.DataFrame, train_ratio_in_tv: float = 1
    ) -> "ModelBase":
        # ensemble don't use train_ratio_in_tv in this function
        self.config.data_dim = train_val_data.shape[1]
        models = get_ensemble_models(self.config.models)
        if _get_val_samples_num(
                len(train_val_data),
                self.config.ensemble_train_ratio_in_tv,
                self.config.seq_len,
                self.config.horizon) <= self.config.min_val_samples_num:
            logger.warning("Too few validation samples, ensemble training skipped")
            for i, model in enumerate(models):
                try:
                    self.best_model = _train_model(model, train_val_data, train_ratio_in_tv, self.config.horizon)
                    return self.best_model
                except Exception as e:
                    logger.warning(f"Model {i + 1} failed: {str(e)}")

            logger.error("All models failed, using DefaultModel")
            self.best_model = DefaultModel()
            return self.best_model

        self.trained_models = self._train_models(
            models,
            train_val_data,
            train_ratio_in_tv
        )

        self.available_model_ids = [i for i, model in enumerate(self.trained_models)]
        self.ensemble_model_ids = [i for i, model in enumerate(self.trained_models) if
                                   getattr(model, "model_name", None) in ENSEMBLE_MODELS]

        if self.available_model_ids:
            len_before_val = get_train_len(
                len(train_val_data),
                self.config.ensemble_train_ratio_in_tv,
                self.config.seq_len,
                self.config.horizon,
            )
            val_dataloader = get_ensemble_dataloader(
                data=train_val_data,
                len_before_predict=self.config.seq_len + len_before_val,  # 给naive方法提供之前的点
                seq_len=self.config.seq_len,
                horizon=self.config.horizon,
                batch_size=self.config.batch_size,
                ensemble=self,
                model_ids=self.available_model_ids,
                shuffle=True,
            )

        if len(self.ensemble_model_ids) <= 1:
            self.ensemble_model_ids = []

        start_time = time.time()
        if self.ensemble_model_ids:
            self.ensemble_model = EnsembleModel(
                self.config,
            )
            train_dataloader = get_ensemble_dataloader(
                data=train_val_data,
                len_before_predict=self.config.seq_len,
                seq_len=self.config.seq_len,
                horizon=self.config.horizon,
                batch_size=self.config.batch_size,
                ensemble=self,
                model_ids=self.ensemble_model_ids,
                shuffle=True,
            )
            self.ensemble_model.init_weight(len(self.ensemble_model_ids))
            optimizer = optim.Adam(self.ensemble_model.parameters(), lr=self.config.lr)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            try:
                self.ensemble_model.to(device)
                for epoch in range(self.config.num_epochs):
                    self.ensemble_model.train()
                    for model_inputs, model_predictions, targets in train_dataloader:
                        optimizer.zero_grad()
                        model_inputs = model_inputs.to(device)
                        model_predictions = model_predictions.to(device)
                        targets = targets.to(device)

                        output = self.ensemble_model(model_inputs, model_predictions)
                        loss = self.criterion(output, targets)

                        loss.backward()
                        optimizer.step()
                    if self.config.ensemble_train_ratio_in_tv != 1:
                        valid_loss = self._val(val_dataloader, self.criterion)
                        self.early_stopping(valid_loss, self.ensemble_model)
                        if self.early_stopping.early_stop:
                            break
                    adjust_learning_rate(optimizer, epoch + 1, self.config)
            except Exception as e:
                logger.error(f"Failed to train ensemble model: {str(e)}")

        self.best_model = self._select_best_model(val_dataloader, self.criterion)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Ensemble time: {execution_time:.5f} s")

    def _val(self, val_dataloader, criterion):
        self.ensemble_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_loss = 0
        with torch.no_grad():
            for model_inputs, model_predictions, targets in val_dataloader:
                selected_model_prediction = model_predictions[:, self.ensemble_model_ids, :, :]
                model_inputs = model_inputs.to(device)
                selected_model_prediction = selected_model_prediction.to(device)
                targets = targets.to(device)
                output = self.ensemble_model(model_inputs, selected_model_prediction)
                loss = criterion(output, targets)
                val_loss += loss.item()
        self.ensemble_model.train()
        return val_loss / len(val_dataloader)

    def forecast(self, horizon: int, series: pd.DataFrame) -> np.ndarray:
        series = series.astype("float32")

        model_name = getattr(self.best_model, 'model_name', None)
        if model_name is not None and model_name == "EnsembleModel":
            if self.early_stopping.check_point is not None:
                self.ensemble_model.load_state_dict(self.early_stopping.check_point)
            model_predictions = self.get_models_predicts(
                self.ensemble_model_ids,
                [series],
                horizon,
            )
            inputs = series.values[-self.config.seq_len:]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ensemble_model.to(device)
            self.ensemble_model.eval()
            with torch.no_grad():
                model_predictions = torch.tensor(model_predictions).float()
                inputs = torch.tensor(inputs).float().unsqueeze(0)
                model_predictions = model_predictions.to(device)
                inputs = inputs.to(device)
                output = self.ensemble_model(inputs, model_predictions)

            output = output.cpu().numpy()
            output = output.reshape(-1, output.shape[-1])
        else:
            output = self.best_model.forecast(horizon, series)
        return output

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        model_name = getattr(self.best_model, 'model_name', None)
        if model_name is not None and model_name == "EnsembleModel":
            input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
            inputs = input_data["input"]
            input_index = input_data["time_stamps"]
            series = [pd.DataFrame(inputs[i], index=input_index[i]) for i in range(inputs.shape[0])]
            if self.early_stopping.check_point is not None:
                self.ensemble_model.load_state_dict(self.early_stopping.check_point)
            model_predictions = self.get_models_predicts(
                self.ensemble_model_ids,
                series,
                horizon,
            )
            inputs = inputs[:, -self.config.seq_len:, :]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.ensemble_model.to(device)
            self.ensemble_model.eval()
            with torch.no_grad():
                model_predictions = torch.tensor(model_predictions).float()
                inputs = torch.tensor(inputs).float()
                model_predictions = model_predictions.to(device)
                inputs = inputs.to(device)
                output = self.ensemble_model(inputs, model_predictions)

            output = output.cpu().numpy()
            return output
        else:
            return self.best_model.batch_forecast(horizon, batch_maker, **kwargs)

    def _train_models(self, model_factories: List[ModelFactory], train: pd.DataFrame, train_ratio_in_tv: float) -> List:
        train = train.astype("float32")

        lock = threading.Lock()
        max_workers = len(model_factories) if self.config.max_workers is None else self.config.max_workers
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
        ) as executor:
            trained_models = []
            futures = [
                executor.submit(_train_model, model_factory, train, train_ratio_in_tv, self.config.horizon)
                for model_factory in model_factories
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    model = future.result()
                except Exception as e:
                    import traceback as tb

                    tb.print_exc()
                    logger.info(f"Failed to train model: {e}")
                    continue
                with lock:
                    trained_models.append(model)
        return trained_models

    def _remove_model(self, model_id):
        if model_id in self.available_model_ids:
            self.trained_models[model_id] = None
            self.available_model_ids.remove(model_id)

        if model_id in self.ensemble_model_ids:
            self.ensemble_model_ids.remove(model_id)

    def get_models_predicts(
            self,
            model_ids: List[int],
            seq_list: List[pd.DataFrame],
            horizon: int,
    ) -> np.ndarray:
        available_model_ids = [i for i in model_ids.copy() if i in self.available_model_ids]
        models = [self.trained_models[i] for i in available_model_ids]
        lock = threading.Lock()
        predicts_list = [None] * len(models)
        max_workers = len(models) if self.config.max_workers is None else self.config.max_workers
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
        ) as executor:
            futures = {
                executor.submit(_get_model_predicts, model_factory, horizon, seq_list): idx
                for idx, model_factory in enumerate(models)
            }

            for future in concurrent.futures.as_completed(futures):
                try:
                    predicts = future.result()
                    idx = futures[future]
                    with lock:
                        predicts_list[idx] = predicts
                except Exception as e:
                    import traceback as tb

                    tb.print_exc()
                    logger.error(str(e))
                    continue

        valid_predicts = []
        for i, pred in enumerate(predicts_list):
            if pred is not None and isinstance(pred, np.ndarray) and np.isfinite(pred).all():
                valid_predicts.append(pred)
            else:
                logger.warning(
                    f"Model {getattr(models[i], 'model_name', type(models[i]).__name__)}'s prediction is abnormal, deleted")
                self._remove_model(available_model_ids[i])

        predicts_list = valid_predicts
        return np.stack(predicts_list, axis=1)

    def _select_best_model(self, val_dataloader: DataLoader, criterion):
        if not self.available_model_ids:
            return DefaultModel()

        if not self.config.select_best_model:
            return self.ensemble_model

        available_models = [self.trained_models[i] for i in self.available_model_ids]
        average_loss = self._get_models_val_loss(val_dataloader, criterion)

        result_record = {}
        for i in range(len(available_models)):
            try:
                result_record[
                    getattr(available_models[i], "model_name", type(available_models[i]).__name__)
                ] = average_loss[i]
            except Exception as e:
                logger.error(f"Failed to get model name: {str(e)}")
        result_record["ensemble"] = self.early_stopping.val_loss_min
        logger.info(f"<<<|{str(result_record)}|>>>")

        min_loss = min(average_loss)
        if min_loss < self.early_stopping.val_loss_min:
            min_index = average_loss.index(min_loss)
            best_model = available_models[min_index]
        else:
            best_model = self.ensemble_model
        return best_model

    def _get_models_val_loss(self, val_dataloader: DataLoader, criterion):
        inputs, models_predicts, targets = val_dataloader.dataset.X, val_dataloader.dataset.Y, val_dataloader.dataset.Z
        all_loss = []
        for models_predict, target in zip(models_predicts, targets):
            models_loss = []
            for model_predict in models_predict:
                loss = criterion(torch.tensor(model_predict), torch.tensor(target))
                models_loss.append(loss.item())

            all_loss.append(models_loss)
        average_loss = np.mean(np.array(all_loss), axis=0).tolist()

        return average_loss
