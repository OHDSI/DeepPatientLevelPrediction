# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn

from Dataset import FeatureInfo
from Embeddings import Embedding


class CustomEmbeddings(nn.Module):
    def __init__(
        self,
        custom_embedding_weights: torch.Tensor,
        embedding_dim: int,
        feature_info: FeatureInfo,
        custom_indices: torch.Tensor,
        freeze: bool = True,
        aggregate: str = "none",
    ):
        """
        Custom embeddings class that allows for the use of custom embeddings
        in addition to regular embeddings. The custom embeddings are
        initialized with the provided weights and can be either trainable
        or frozen. The regular embeddings are initialized with random
        weights.

        Args:
            custom_embedding_weights: shape N x D where N is the number of
                custom embeddings and D is the embedding dimension
            embedding_dim: the dimension of the regular embeddings
            num_regular_embeddings: the number of regular embeddings
            custom_indices: indices of the custom embeddings in the regular
                vocabulary
            freeze: whether to freeze the custom embeddings or not

        Returns:
            A CustomEmbeddings object
        """
        super(CustomEmbeddings, self).__init__()
        num_custom_embeddings = custom_embedding_weights.shape[0]
        # make sure padding idx refers to all zero embeddings at position 0
        custom_embedding_weights = torch.cat(
            [
                torch.zeros(1, custom_embedding_weights.shape[1]),
                custom_embedding_weights,
            ]
        )
        self.custom_embeddings = nn.Embedding.from_pretrained(
            custom_embedding_weights, freeze=freeze, padding_idx=0
        )
        self.custom_embeddings_trainable = nn.Embedding(
            num_custom_embeddings + 1, custom_embedding_weights.shape[-1], padding_idx=0
        )
        assert (
            self.custom_embeddings_trainable.weight.shape
            == custom_embedding_weights.shape
        )
        nn.init.zeros_(
            self.custom_embeddings_trainable.weight
        )  # initialize trainable embeddings to 0
        num_regular_embeddings = feature_info.get_vocabulary_size()
        self.aggregate = aggregate

        self.regular_embeddings = Embedding(embedding_dim, 
                                            feature_info=feature_info,
                                            aggregate=aggregate)

        self.register_buffer("custom_indices", custom_indices)

        if custom_embedding_weights.shape[1] != embedding_dim:
            self.linear_transform = nn.Linear(
                custom_embedding_weights.shape[1], embedding_dim
            )
        else:
            self.linear_transform = nn.Identity()

        # create a tensor that such that tensor[input] will give the index of 
        # the custom embedding in self.custom_embeddings
        # the first index is 0, which is the padding index
        vocab_to_custom = torch.zeros(num_regular_embeddings + 1, dtype=torch.long)
        vocab_to_custom[custom_indices] = torch.arange(1, custom_indices.shape[0] + 1)
        self.register_buffer("vocab_to_custom", vocab_to_custom)

        vocab_to_regular = torch.zeros(num_regular_embeddings + 1, dtype=torch.long)
        regular_indices = torch.where(vocab_to_custom == 0)[0]
        vocab_to_regular[regular_indices] = torch.arange(
            0, num_regular_embeddings - custom_indices.shape[0] + 1
        )
        self.register_buffer("vocab_to_regular", vocab_to_regular)

    @staticmethod
    def process_custom_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings  # default is identity for Euclidian embeddings

    def forward(self, x):
        # count before the feature ids are split into custom and regular below
        sequence_lengths = (x["feature_ids"] != 0).sum(dim=1, keepdim=True).clamp_min(1)
        custom_embeddings_mask = torch.isin(x["feature_ids"],
                                            self.custom_indices.to(x["feature_ids"].device))
        custom_features = torch.where(custom_embeddings_mask, x["feature_ids"], torch.tensor(0))
        x["feature_ids"] = torch.where(~custom_embeddings_mask, x["feature_ids"], torch.tensor(0))
        x["feature_values"] = torch.where(~custom_embeddings_mask, x["feature_values"], torch.tensor(0.0))
        custom_mapped_features = self.vocab_to_custom[custom_features]
        custom_embeddings = self.custom_embeddings(custom_mapped_features)
        custom_embeddings = self.process_custom_embeddings(custom_embeddings)
        custom_embeddings = custom_embeddings + self.custom_embeddings_trainable(
            custom_mapped_features
        )
        # the projection applies to custom features only, so that its bias is
        # not added for padding or regular features
        mask = custom_embeddings_mask.unsqueeze(-1).to(dtype=custom_embeddings.dtype)
        regular_embeddings = self.regular_embeddings(x)
        if self.aggregate == "sum":
            custom_embeddings = self.project_sum(custom_embeddings * mask, mask)
            sequence_lengths = sequence_lengths.to(dtype=custom_embeddings.dtype)
            return (custom_embeddings + regular_embeddings) / sequence_lengths
        else:
            return self.linear_transform(custom_embeddings) * mask + regular_embeddings

    def project_sum(self, custom_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sum the projected custom features of each patient.

        The projection is linear, so this projects the sum and adds the bias
        once per custom feature, without a tensor of the model's width for
        every position.
        """
        summed = custom_embeddings.sum(dim=1)
        if not isinstance(self.linear_transform, nn.Linear):
            return self.linear_transform(summed)
        projected = nn.functional.linear(summed, self.linear_transform.weight)
        if self.linear_transform.bias is not None:
            projected = projected + mask.sum(dim=1) * self.linear_transform.bias
        return projected


def check_poincare_points(points: torch.Tensor, curvature: float) -> None:
    """Check that the points lie inside the Poincare ball with curvature -c.

    Points of that ball have a norm below 1 / sqrt(c).
    """
    if not math.isfinite(curvature) or curvature <= 0:
        raise ValueError(f"The curvature must be a positive number, not {curvature}")
    if not torch.isfinite(points).all():
        raise ValueError("The Poincare embeddings contain values that are not finite")
    norms = points.detach().double().norm(dim=-1)
    outside = norms >= 1.0 / math.sqrt(curvature)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} of {norms.numel()} Poincare embeddings are not "
            f"inside the Poincare ball with curvature {curvature:g}, where the norm "
            f"must be below 1 / sqrt({curvature:g}) = {1.0 / math.sqrt(curvature):.4g}. "
            f"The largest norm is {norms.max().item():.4g}. Check the curvature the "
            "embeddings were trained with. A model saved by an earlier version with "
            "these embeddings mapped them wrongly and needs retraining."
        )


def logmap0(points: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    """Map points of the Poincare ball with curvature -c to the tangent space
    at the origin.

    Near the origin the scale factor comes from its series, which keeps the
    result and its gradient finite at zero, the padding embedding.
    """
    # scale before squaring, so that a small curvature cannot overflow
    z = (math.sqrt(curvature) * points).square().sum(dim=-1, keepdim=True)
    # the series 1 + z/3 + z^2/5 is off by about z^3/7, far below float precision
    small = z < 1e-4
    # torch.where also differentiates the branch it does not select, so that
    # branch gets inputs with a finite gradient
    z_exact = torch.where(small, torch.full_like(z, 0.25), z)
    # the largest radius below 1, so that rounding cannot reach the boundary
    # and points inside the ball are not clipped
    below_one = 1.0 - torch.finfo(points.dtype).eps / 2
    radius = z_exact.sqrt().clamp_max(below_one)
    z_series = torch.where(small, z, torch.zeros_like(z))
    scale = torch.where(
        small, 1 + z_series / 3 + z_series.square() / 5, torch.atanh(radius) / radius
    )
    return points * scale


class PoincareEmbeddings(CustomEmbeddings):
    """Custom embeddings that are points in the Poincare ball.

    The points are mapped to the tangent space at the origin, which is a
    Euclidean space. The trainable offsets are vectors in that tangent space
    too, so they are added to the mapped points, as for other custom
    embeddings.

    Args:
        curvature: the c of the ball with curvature -c that the embeddings
            were trained in. Libraries that constrain a curvature parameter,
            such as HypLL with softplus, have an effective c that differs from
            the parameter.
    """

    def __init__(self, custom_embedding_weights: torch.Tensor, *args,
                 curvature: float = 1.0, **kwargs):
        check_poincare_points(custom_embedding_weights, float(curvature))
        super().__init__(custom_embedding_weights, *args, **kwargs)
        if self.custom_embeddings.weight.requires_grad:
            # an optimizer step on the coordinates themselves can move them out
            # of the ball, which needs a Riemannian optimizer to prevent
            raise ValueError(
                "PoincareEmbeddings needs frozen embeddings. Training the points "
                "of the ball with a Euclidean optimizer can move them outside it. "
                "The trainable offsets adapt the embeddings in the tangent space."
            )
        self.curvature = float(curvature)

    def process_custom_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the logmap0 function to the embeddings to project them to the
        tangent space at the origin.

        Args:
            embeddings: shape N x D where N is the number of embeddings and
                D is the embedding dimension

        Returns:
            A logmapped tensor of same shape as the input tensor
        """
        return logmap0(embeddings, self.curvature)
