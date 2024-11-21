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
        num_custom_embeddings = custom_embedding_weights.shape[0]
        # make sure padding idx refers to all zero embeddings at position 0
        custom_embedding_weights = torch.cat([torch.zeros(1, custom_embedding_weights.shape[1]),
                                              custom_embedding_weights])
        self.custom_embeddings = nn.Embedding.from_pretrained(custom_embedding_weights, freeze=freeze,
                                                              padding_idx=0)
        self.custom_embeddings_trainable = nn.Embedding(num_custom_embeddings + 1, custom_embedding_weights.shape[-1], padding_idx=0)
        assert(self.custom_embeddings_trainable.weight.shape == custom_embedding_weights.shape)
        nn.init.zeros_(self.custom_embeddings_trainable.weight) # initialize trainable embeddings to 0

        self.regular_embeddings = nn.Embedding(num_regular_embeddings - num_custom_embeddings, embedding_dim, padding_idx=0)

        self.register_buffer("custom_indices", custom_indices)

        if custom_embedding_weights.shape[1] != embedding_dim:
            self.linear_transform = nn.Linear(custom_embedding_weights.shape[1], embedding_dim)
        else:
            self.linear_transform = nn.Identity()

        # create a tensor that such that tensor[input] will give the index of the custom embedding in self.custom_embeddings
        vocab_to_custom = torch.zeros(num_regular_embeddings, dtype=torch.long)
        vocab_to_custom[custom_indices] = torch.arange(1, custom_indices.shape[0] + 1)
        self.register_buffer("vocab_to_custom", vocab_to_custom)

        vocab_to_regular = torch.zeros(num_regular_embeddings, dtype=torch.long)
        regular_indices = torch.where(vocab_to_custom == 0)[0]
        vocab_to_regular[regular_indices] = torch.arange(0, num_regular_embeddings - num_custom_embeddings)
        self.register_buffer("vocab_to_regular", vocab_to_regular)

    @staticmethod
    def process_custom_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings # default is identity for Euclidian embeddings

    def combine_embeddings(self, fixed_embeddings: torch.Tensor, trainable_embeddings: torch.Tensor) -> torch.Tensor:
        return fixed_embeddings + trainable_embeddings

    def forward(self, x):
        custom_embeddings_mask = torch.isin(x, self.custom_indices.to(x.device))
        custom_features = torch.where(custom_embeddings_mask, x, torch.tensor(0))
        regular_features = torch.where(~custom_embeddings_mask, x, torch.tensor(0))
        custom_mapped_features = self.vocab_to_custom[custom_features]
        custom_embeddings = self.custom_embeddings(custom_mapped_features)
        custom_embeddings = self.process_custom_embeddings(custom_embeddings)
        custom_trainable_embeddings = self.custom_embeddings_trainable(custom_mapped_features)
        custom_embeddings = self.combine_embeddings(custom_embeddings, custom_trainable_embeddings)
        # custom_embeddings = self.process_custom_embeddings(custom_embeddings)
        regular_embeddings = self.regular_embeddings(self.vocab_to_regular[regular_features])
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
    def process_custom_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the logmap0 function to the embeddings to project them to the tangent space at the origin.
        Parameters
        ----------
        embeddings

        Returns
        -------

        """
        return logmap0(embeddings)

    def combine_embeddings(self, fixed_embeddings: torch.Tensor, trainable_embeddings: torch.Tensor) -> torch.Tensor:
        return self.mobius_add(fixed_embeddings, trainable_embeddings)

    def mobius_add(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Perform Mobius addition of two embeddings.
        Parameters
        ----------
        embedding1          shape N x D where N is the number of embeddings and D is the embedding dimension
        embedding2          shape N x D where N is the number of embeddings and D is the embedding dimension

        Returns
        -------

        """
        x2 = embedding1.pow(2).sum(dim=-1, keepdim=True)
        y2 = embedding2.pow(2).sum(dim=-1, keepdim=True)
        xy = (embedding1 * embedding2).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * embedding1 + (1 - x2) * embedding2
        denom = 1 + 2 * xy + x2 * y2
        return num / denom.clamp_min(1e-15)


