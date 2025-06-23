import copy
import logging

import numpy as np

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Class to implement early stopping during training, to prevent overfitting by halting training
    when the validation loss stops improving for a specified number of epochs (patience).
    """

    def __init__(self, patience=7, delta=0):
        """
        Initializes the early stopping criteria.

        :param patience: The number of epochs to wait for an improvement in validation loss before stopping.
        :param delta: The minimum change in validation loss to count as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.check_point = None

    def __call__(self, val_loss, model):
        """
        This method is called after each validation step to check if the validation loss has improved.

        If the validation loss does not improve for 'patience' number of epochs, early stopping is triggered.

        :param val_loss: The current validation loss.
        :param model: The model being trained.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves the model checkpoint when the validation loss improves.

        :param val_loss: The validation loss when the improvement occurred.
        :param model: The model whose state is being saved.
        """
        logger.info(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
        )
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
