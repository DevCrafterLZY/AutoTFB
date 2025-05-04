import torch
import torch.nn as nn

from AutoTFB.models.layers.mlp import MLP


class NCF(nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()

        self.feature_dim = config.feature_dim1 * config.feature_dim2
        self.hidden_dims = config.hidden_dims
        self.dropout = config.dropout
        self.output_dim = config.output_dim
        self.model_num = config.model_num

        self.tsvec_dim = config.tsvec_dim
        self.model_dim = config.model_dim

        self.tsvec_encoder = MLP(input_dim=self.feature_dim, hidden_dims=self.hidden_dims, output_dim=self.tsvec_dim,
                                 dropout=self.dropout)
        self.model_embedding = nn.Embedding(self.model_num, self.model_dim)
        self.interaction_layer = nn.Sequential(
            nn.Linear(self.tsvec_dim + self.model_dim, (self.tsvec_dim + self.model_dim) // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.output_layer = nn.Sequential(
            nn.Linear((self.tsvec_dim + self.model_dim) // 2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, i):
        """
        :param x: dataset feature (batch_size, feature_dim1)
        :param i: model embedding (batch_size, )
        :return: prediction (batch_size, output_dim)
        """
        tsvec = self.tsvec_encoder(x)  # (batch_size, tsvec_dim)
        model_emb = self.model_embedding(i)  # (batch_size, model_dim)

        combined_emb = torch.cat((tsvec, model_emb), dim=-1)  # (batch_size, tsvec_dim + model_dim)
        interaction_output = self.interaction_layer(combined_emb)  # (batch_size, hidden_dim)
        output = self.output_layer(interaction_output)  # (batch_size, output_dim)

        return output

    def get_combined_emb(self, x):
        """
        :param x: dataset feature (1, feature_dim1)
        :return: combined embedding (batch_size, tsvec_dim + model_dim)
        """
        tsvec = self.tsvec_encoder(x)
        tsvec_expanded = tsvec.expand(self.model_num, -1)
        model_emb = self.model_embedding.weight

        combined_emb = torch.cat((tsvec_expanded, model_emb), dim=-1)
        return combined_emb
