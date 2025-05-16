import torch
from torch import nn

from inst.python.Dataset import FeatureInfo
from inst.python.Embeddings import NumericalEmbedding


class ResNet(nn.Module):
    def __init__(
        self,
        feature_info: FeatureInfo,
        size_embedding: int = 256,
        size_hidden: int = 256,
        num_layers: int = 2,
        hidden_factor: int = 1,
        activation=nn.ReLU,
        normalization=nn.BatchNorm1d,
        hidden_dropout=0,
        residual_dropout=0,
        dim_out: int = 1,
        concat_num=True,
        model_type="ResNet",
    ):
        super(ResNet, self).__init__()
        self.name = model_type
        size_embedding = int(size_embedding)
        size_hidden = int(size_hidden)
        num_layers = int(num_layers)
        hidden_factor = int(hidden_factor)
        dim_out = int(dim_out)

        self.embedding = ResNetEmbedding(
            feature_info=feature_info,
            concat_num=concat_num,
            embedding_dim=size_embedding
        )

        # self.embedding = nn.EmbeddingBag(
        #     num_embeddings=cat_features + 1, embedding_dim=size_embedding, padding_idx=0
        # )
        # if num_features != 0 and not concat_num:
        #     self.num_embedding = NumericalEmbedding(num_features, size_embedding)
        # else:
        #     self.num_embedding = None
        #     size_embedding = size_embedding + num_features

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        res_hidden = size_hidden * hidden_factor

        self.layers = nn.ModuleList(
            ResLayer(
                size_hidden,
                res_hidden,
                normalization,
                activation,
                hidden_dropout,
                residual_dropout,
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


class ResLayer(nn.Module):
    def __init__(
        self,
        size_hidden,
        res_hidden,
        normalization,
        activation,
        hidden_dropout=None,
        residual_dropout=None,
    ):
        super(ResLayer, self).__init__()

        self.norm = normalization(size_hidden)
        self.linear0 = nn.Linear(size_hidden, res_hidden)
        self.linear1 = nn.Linear(res_hidden, size_hidden)

        if hidden_dropout is not None:
            self.hidden_dropout = nn.Dropout(p=hidden_dropout)
        if residual_dropout is not None:
            self.residual_dropout = nn.Dropout(p=residual_dropout)
        self.activation = activation()

    def forward(self, input):
        z = input
        z = self.norm(z)
        z = self.linear0(z)
        z = self.activation(z)
        if self.hidden_dropout is not None:
            z = self.hidden_dropout(z)
        z = self.linear1(z)
        if self.residual_dropout is not None:
            z = self.residual_dropout(z)
        z = z + input
        return z


class ResNetEmbedding(nn.Module):
    def __init__(
        self, feature_info: FeatureInfo, concat_num: bool = False, embedding_dim: int = 128
    ) -> None:
        super().__init__()
        self.concat_num = concat_num
        self.numerical_feature_ids = feature_info.get_numerical_feature_ids()
        self.numerical_embeddings = self.numerical_feature_ids.shape[0]
        mode = "concatenate" if concat_num else "scale"
        self.numerical_embedding = NumericalEmbedding(num_embeddings=self.numerical_embeddings,
                                                      embedding_dim=embedding_dim,
                                                      mode=mode,
                                                      aggregate=True)
        self.vocabulary_size = feature_info.get_vocabulary_size()
        categorical_embedding_size = self.vocabulary_size - self.numerical_embeddings
        self.categorical_embedding = nn.EmbeddingBag(num_embeddings=categorical_embedding_size + 1,
                                                     embedding_dim=embedding_dim,
                                                     padding_idx=0,
                                                     mode = "sum")

        input_to_numeric = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        input_to_numeric[self.numerical_feature_ids] = torch.arange(
            1, self.numerical_feature_ids.shape[0] + 1
        )
        self.register_buffer("input_to_numeric", input_to_numeric)

        input_to_categorical = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        categorical_feature_ids = torch.where(input_to_numeric == 0)[0]
        input_to_categorical[categorical_feature_ids[1:]] = torch.arange(
            1, categorical_feature_ids.numel()
        )
        self.register_buffer("input_to_categorical", input_to_categorical)

    def forward(self, x: dict) -> torch.Tensor:
        """

        Args:
            x (dict): A dictionary containing the input data. The keys should be:
                        - "feature_ids": A tensor feature_ids.
                        - "feature_values": A tensor of numerical features (optional).
        Returns:
            torch.Tensor: The output embeddings.
        """
        numerical_mask = torch.isin(
            x["feature_ids"],
            self.numerical_feature_ids.to(x["feature_ids"].device)
        )
        numerical_features = torch.where(
            numerical_mask, x["feature_ids"],
            torch.tensor(0)
        )
        numerical_mapped_features = self.input_to_numeric[numerical_features]
        numerical_values = torch.where(
            numerical_mask, x["feature_values"], torch.tensor(0.0)
        )
        numerical_embeddings = self.numerical_embedding(
            numerical_mapped_features, numerical_values
        )
        categorical_features = torch.where(
            ~numerical_mask, x["feature_ids"], torch.tensor(0)
        )
        categorical_mapped_features = self.input_to_categorical[categorical_features]
        categorical_embeddings = self.categorical_embedding(categorical_mapped_features)
        merged_embeddings = (categorical_embeddings + numerical_embeddings) / numerical_mask.shape[1]
        return merged_embeddings
