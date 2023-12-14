from torch import nn

from ResNet import NumericalEmbedding


class MLP(nn.Module):
    def __init__(
        self,
        cat_features: int,
        num_features: int,
        size_embedding: int,
        size_hidden: int,
        num_layers: int,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
        dropout=None,
        dim_out: int = 1,
        model_type="MLP"
    ):
        super(MLP, self).__init__()
        self.name = model_type
        cat_features = int(cat_features)
        num_features = int(num_features)
        size_embedding = int(size_embedding)
        size_hidden = int(size_hidden)
        num_layers = int(num_layers)
        dim_out = int(dim_out)

        self.embedding = nn.EmbeddingBag(
            cat_features + 1, size_embedding, padding_idx=0
        )

        if num_features != 0 and num_features is not None:
            self.num_embedding = NumericalEmbedding(num_features, size_embedding)

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        self.layers = nn.ModuleList(
            MLPLayer(
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

    def forward(self, input):
        x_cat = input["cat"]
        x_cat = self.embedding(x_cat)
        if "num" in input.keys() and self.num_embedding is not None:
            x_num = input["num"]
            x = (x_cat + self.num_embedding(x_num).mean(dim=1)) / 2
        else:
            x = x_cat
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


class MLPLayer(nn.Module):
    def __init__(
        self,
        size_hidden=64,
        normalization=nn.BatchNorm1d,
        activation=nn.ReLU,
        dropout=0.0,
        bias=True,
    ):
        super(MLPLayer, self).__init__()
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
