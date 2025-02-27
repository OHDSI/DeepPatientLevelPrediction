import math

import torch
from torch import nn
import polars as pl

class Embedding(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 feature_info: dict):
        super(Embedding, self).__init__()
        self.embedding_dim = int(embedding_dim)

        self.vocabulary_size = (
            feature_info["data_reference"]
            .filter(pl.col("isBinary") == "Y")
            .select("columnId")
            .max()
            .item()
        )
        self.numerical_feature_ids = (
            feature_info["data_reference"]
            .filter(pl.col("isBinary") == "N")
            .select("columnId")
            .to_torch()
        )

        self.embedding = nn.Embedding(
            self.vocabulary_size + 1 - self.numerical_feature_ids.shape[0], embedding_dim, padding_idx=0)

        if self.numerical_feature_ids.shape[0] != 0:
            self.numerical_embedding = (
                NumericalEmbedding(self.numerical_feature_ids.shape[0],
                                   embedding_dim)
            )

        # create a router to router the input to the correct embedding such that
        # input_to_numeric[input] will give the index of the numerical feature in numerical_embedding
        input_to_numeric = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        input_to_numeric[self.numerical_feature_ids] = torch.arange(1, self.numerical_feature_ids.shape[0] + 1)
        self.register_buffer("input_to_numeric", input_to_numeric)

        input_to_categorical = torch.zeros(self.vocabulary_size + 1, dtype=torch.long)
        categorical_feature_ids = torch.where(input_to_numeric == 0)[0]
        input_to_categorical[categorical_feature_ids[1:]] = torch.arange(1, categorical_feature_ids.numel())
        self.register_buffer("input_to_categorical", input_to_categorical)

    def forward(self, x):
        numerical_mask = torch.isin(x["feature_ids"], self.numerical_feature_ids.to(x["feature_ids"].device))
        numerical_features = torch.where(numerical_mask, x["feature_ids"], torch.tensor(0))
        numerical_mapped_features = self.input_to_numeric[numerical_features]
        numerical_values = x["feature_values"][numerical_mask]
        numerical_embeddings = self.numerical_embedding(numerical_mapped_features, numerical_values)
        categorical_features = torch.where(~numerical_mask, x["feature_ids"], torch.tensor(0))
        categorical_mapped_features = self.input_to_categorical[categorical_features]
        categorical_embeddings = self.embedding(categorical_mapped_features)
        merge_embeddings = torch.where(numerical_mask.unsqueeze(-1), numerical_embeddings, categorical_embeddings)
        return merge_embeddings


class NumericalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, bias=True):
        super(NumericalEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        else:
            self.bias = None

        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))

    def forward(self, *inputs):
        values = inputs[1]# only use values in this one
        x = self.weight * values.unsqueeze(1)
        if self.bias is not None:
            x = x.unsqueeze(1) + self.bias[None]
        return x


class NumericalEmbedding2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(NumericalEmbedding2, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim - 1, padding_idx=0)

    def forward(self, ids, values):
        x = self.embedding(ids)
        return torch.cat((x, values[:, None, None].expand(-1, x.shape[1], -1)), dim=-1)


class ClassTokenNested(nn.Module):
        """
        Prepend a class token to the input tensor
        """

        def __init__(self, dim_token):
            super(ClassToken, self).__init__()
            self.weight = nn.Parameter(torch.empty(1, dim_token))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


        @torch._dynamo.disable
        def add_token(self, x):
            return torch.nested.as_nested_tensor(
                [torch.cat([self.weight, seq]) for seq in x.unbind(0)],
                device=x.device,
                layout=x.layout)

        def forward(self, x):
            return self.add_token(x)


class ClassToken(nn.Module):
    def __init__(self, dim_token):
        super(ClassToken, self).__init__()
        self.weight = nn.Parameter(torch.empty(dim_token, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def expand(self, dims):
        new_dims = [1] * (len(dims) - 1)
        return self.weight.view(new_dims + [-1]).expand(dims + [-1])

    def forward(self, x):
        return torch.cat([self.expand([x.shape[0], 1]), x], dim=1)

