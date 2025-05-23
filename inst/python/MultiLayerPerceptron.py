from torch import nn

from Dataset import FeatureInfo
from Embeddings import Embedding


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        feature_info: FeatureInfo,
        size_embedding: int,
        size_hidden: int,
        num_layers: int,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
        dropout=0.0,
        dim_out: int = 1,
        concat_num=True,
        model_type="MultiLayerPerceptron"
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.name = model_type
        size_embedding = int(size_embedding)
        size_hidden = int(size_hidden)
        num_layers = int(num_layers)
        dim_out = int(dim_out)


        self.embedding = Embedding(
            feature_info=feature_info,
            numeric_mode="concatenate" if concat_num else "scale",
            embedding_dim=size_embedding,
            aggregate="sum"
        )

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        self.layers = nn.ModuleList(
            MlpLayer(
                size_hidden=size_hidden,
                normalization=normalization,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.last_norm = normalization(size_hidden)
        self.head = nn.Linear(size_hidden, dim_out)
        self.size_hidden = size_hidden
        self.dim_out = dim_out

        self.last_act = activation()

    def forward(self, x):
        x = self.embedding(x)
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_norm(x)
        x = self.last_act(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def reset_head(self):
        self.head = nn.Linear(self.size_hidden, self.dim_out)


class MlpLayer(nn.Module):
    def __init__(
        self,
        size_hidden=64,
        normalization=nn.BatchNorm1d,
        activation=nn.ReLU,
        dropout=0.0,
        bias=True,
    ):
        super(MlpLayer, self).__init__()
        self.norm = normalization(size_hidden)
        self.activation = activation()
        self.linear = nn.Linear(size_hidden, size_hidden, bias=bias)

        if dropout != 0.0 or dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        return self.activation(x)
