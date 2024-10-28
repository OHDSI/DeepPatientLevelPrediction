import torch
import torch.nn as nn

class CustomEmbeddings(nn.Module):
    def __init__(self,
                 custom_embedding_weights: torch.Tensor,
                 embedding_dim: int,
                 num_regular_embeddings: int,
                 custom_indices: torch.Tensor,
                 freeze: bool = True):
        super(CustomEmbeddings, self).__init__()

        # make sure padding idx refers to all zero embeddings at position 0
        custom_embedding_weights = torch.cat([torch.zeros(1, custom_embedding_weights.shape[1]), custom_embedding_weights])
        self.custom_embeddings = nn.Embedding.from_pretrained(custom_embedding_weights, freeze=freeze,
                                                              padding_idx=0)
        self.regular_embeddings = nn.Embedding(num_regular_embeddings, embedding_dim, padding_idx=0)

        self.custom_indices = custom_indices

        self.linear_transform = nn.Linear(custom_embedding_weights.shape[1], embedding_dim)

    @staticmethod
    def process_custom_embeddings(embeddings: torch.Tensor):
        return embeddings

    def forward(self, x):
        custom_embeddings_mask = torch.isin(x, self.custom_indices.to(x.device))
        custom_features = torch.where(custom_embeddings_mask, x, torch.tensor(0))
        regular_features = torch.where(~custom_embeddings_mask, x, torch.tensor(0))

        custom_embeddings = self.custom_embeddings(custom_features)
        custom_embeddings = self.process_custom_embeddings(custom_embeddings)
        regular_embeddings = self.regular_embeddings(regular_features)

        custom_embeddings = self.linear_transform(custom_embeddings)

        return custom_embeddings + regular_embeddings

def logmap0(input_tensor: torch.Tensor):
    curvature = 1.0
    norm_input = torch.norm(input_tensor, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(torch.tensor(curvature, dtype=input_tensor.dtype, device=input_tensor.device))
    scale = torch.arctanh(sqrt_c * norm_input) / (sqrt_c * norm_input)
    scale[torch.isnan(scale)] = 1.0
    return scale * input_tensor


class PoincareEmbeddings(CustomEmbeddings):
    @staticmethod
    def process_custom_embeddings(embeddings: torch.Tensor):
        return logmap0(embeddings)
