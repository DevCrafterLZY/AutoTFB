import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader

from AutoTFB.models.model import NCF
from AutoTFB.utils.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


def adjust_learning_rate(optimizer, epoch, lr):
    """
    Adjusts the learning rate during training

    :param optimizer: The optimizer whose learning rate is to be adjusted.
    :param epoch: The current epoch number, used to determine the adjustment.
    :param lr: The initial learning rate before any adjustments.
    :return: None
    """
    adjust_lr = lr * (0.5 ** ((epoch - 1) // 1))
    for params in optimizer.param_groups:
        params['lr'] = adjust_lr
    logger.info('Updating learning rate to {:.3g}'.format(adjust_lr))


class NCF_model:
    """
    A class for training and testing the model.

    This class handles the model training, testing, and early stopping based on validation loss.
    """

    def __init__(self, config):
        self.config = config
        self.model = NCF(config)
        self.early_stopping = EarlyStopping(patience=config.patience)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def train(self, train_loader: dataloader, val_loader: dataloader):
        """
        Trains the model on the provided training data.

        This method performs the forward pass, computes the loss, applies backpropagation,
        and updates the model's parameters for each epoch.

        :param train_loader: DataLoader for the training dataset.
        :param val_loader: DataLoader for the validation dataset.
        :return: None
        """
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f'Total number of parameters: {total_params}')
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_train_loss = 0

            for tsvec_batch, model_id_batch, target_batch in train_loader:
                tsvec_batch = tsvec_batch.to(self.device)
                model_id_batch = model_id_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                output = self.model(tsvec_batch, model_id_batch)

                loss = self.criterion(output, target_batch)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            adjust_learning_rate(optimizer, epoch + 1, self.config.lr)
            avg_loss = total_train_loss / len(train_loader)

            val_loss = self._val(val_loader)
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                break
            logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}], "
                        f"Loss: {avg_loss:.4f}, "
                        f"val Loss: {val_loss:.4f}")

    def _val(self, val_loader: dataloader):
        """
        Evaluate the model on the validation set.

        :param val_loader: DataLoader for the validation dataset.
        :return: validation loss
        """
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tsvec_batch, model_id_batch, target_batch in val_loader:
                tsvec_batch = tsvec_batch.to(self.device)
                model_id_batch = model_id_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                output = self.model(tsvec_batch, model_id_batch)

                loss = self.criterion(output, target_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        self.model.train()
        return avg_val_loss

    def test(self, test_loader: dataloader) -> (np.ndarray, np.ndarray):
        """
        Tests the model on the provided test dataset.

        This method evaluates the model on the test data without performing backpropagation.

        :param test_loader: DataLoader for the test dataset.
        :return: The predictions and targets for the test dataset.
        """
        if self.early_stopping.check_point is not None:
            self.model.load_state_dict(self.early_stopping.check_point)
            logger.info("Loading the best model.")
        self.model.to(self.device)
        self.model.eval()

        all_predictions = []
        all_targets = []

        total_test_loss = 0
        with torch.no_grad():
            for tsvec_batch, model_id_batch, target_batch in test_loader:
                tsvec_batch = tsvec_batch.to(self.device)
                model_id_batch = model_id_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                output = self.model(tsvec_batch, model_id_batch)

                all_predictions.append(output.cpu().numpy())
                all_targets.append(target_batch.cpu().numpy())

                loss = self.criterion(output, target_batch)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")

        return np.vstack(all_predictions), np.vstack(all_targets)
