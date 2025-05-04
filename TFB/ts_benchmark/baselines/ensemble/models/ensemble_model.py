import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


def msmape_loss(input, target, epsilon=0.1):
    comparator = torch.full_like(target, 0.5 + epsilon)
    denom = torch.maximum(comparator, torch.abs(input) + torch.abs(target) + epsilon)
    msmape_per_series = torch.mean(2 * torch.abs(input - target) / denom) * 100
    return msmape_per_series


class EnsembleModel(nn.Module):
    def __init__(
            self,
            config,
    ):
        super().__init__()

        self.prediction_horizon = config.horizon
        self.model_seq_len = config.seq_len
        self.data_dim = config.data_dim
        self.strategy = config.strategy
        self.model_name = "EnsembleModel"

    def init_weight(self, weight_num: int, dim_num: int = 1):
        if self.strategy == "average":
            self.weight = nn.Parameter(
                torch.full(
                    (dim_num, weight_num),
                    1 / weight_num,
                    dtype=torch.float32
                ),
                requires_grad=False
            )
        elif self.strategy == "weighted":
            self.weighting_network = nn.Sequential(
                nn.Linear(self.data_dim * self.model_seq_len, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, weight_num)  # 输出一个标量作为样本的权重
            )
        else:
            raise ValueError("Unsupported ensemble method")

    def forward(self, model_inputs: torch.Tensor, model_predictions: torch.Tensor):
        """
        :param model_predictions: (n, j, k, l)
            n: batch_size
            j: num_models
            k: time_stamps
            l: series_dim
        """
        n, k, l = model_inputs.shape
        model_inputs_reshaped = model_inputs.reshape(n, k * l)
        sample_weights = self.weighting_network(model_inputs_reshaped)
        softmax_weights = F.softmax(sample_weights, dim=1)
        # if self.training:
        #     softmax_weights = F.softmax(sample_weights, dim=1)
        # else:
        #     max_indices = torch.argmax(sample_weights, dim=1, keepdim=True)
        #     softmax_weights = torch.zeros_like(sample_weights)
        #     softmax_weights.scatter_(1, max_indices, 1)

        weighted_predict = torch.einsum("nj,njkl->nkl", softmax_weights, model_predictions)
        return weighted_predict
