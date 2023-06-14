import math

import torch
from torch import nn


class ResNet(nn.Module):

    def __init__(self,
                 cat_features,
                 num_features=0,
                 size_embedding=256,
                 size_hidden=256,
                 num_layers=2,
                 hidden_factor=1,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm1d,
                 hidden_dropout=0,
                 residual_dropout=0,
                 dim_out=1,
                 concat_num=True):
        super(ResNet, self).__init__()
        self.name = 'ResNet'
        self.embedding = nn.EmbeddingBag(
            num_embeddings=cat_features,
            embedding_dim=size_embedding,
            padding_idx=0
        )
        if num_features != 0 and concat_num:
            self.num_embedding = NumericalEmbedding(num_features, size_embedding)
        else:
            self.num_embedding = None
            size_embedding = size_embedding + num_features

        self.first_layer = nn.Linear(size_embedding, size_hidden)

        res_hidden = size_hidden * hidden_factor

        self.layers = nn.ModuleList(ResLayer(size_hidden, res_hidden, normalization,
                                             activation, hidden_dropout, residual_dropout)
                                    for _ in range(num_layers))

        self.last_norm = normalization(size_hidden)

        self.head = nn.Linear(size_hidden, dim_out)

        self.last_act = activation()

    def forward(self, x):
        x_cat = x['cat']
        x_num = x['num']
        x_cat = self.embedding(x_cat)
        if x_num is not None and self.num_embedding is not None:
            # take the average af numerical and categorical embeddings
            x = (x_cat + self.num_embedding(x_num).mean(dim=1)) / 2
        elif x_num is not None and self.num_embedding is None:
            # concatenate numerical to categorical embedding
            x = torch.cat([x_cat, x_num], dim=1)
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


class ResLayer(nn.Module):

    def __init__(self,
                 size_hidden,
                 res_hidden,
                 normalization,
                 activation,
                 hidden_dropout=None,
                 residual_dropout=None):
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


class NumericalEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 bias=False):
        super(NumericalEmbedding, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        else:
            self.bias = None

        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))

    def forward(self, input):
        x = self.weight.unsqueeze(0) * input.unsqueeze(-1)
        if self.bias is not None:
            x = x + self.bias.unsqueeze(-1)
        return x



