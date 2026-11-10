# Copyright 2022-2026 Observational Health Data Sciences and Informatics
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import pathlib

import torch
import polars as pl

from CustomEmbeddings import CustomEmbeddings, PoincareEmbeddings
from PositionalEncodings import create_positional_encoding_module


class InitStrategy(ABC):
    @abstractmethod
    def initialize(self, model, parameters):
        pass


class DefaultInitStrategy(InitStrategy):
    def initialize(self, model, parameters):
        if "positional_encoding" in parameters["model_parameters"]:
            parameters["model_parameters"]["pe_module"] = (
                create_positional_encoding_module(parameters["model_parameters"])
            )
        return model(**parameters["model_parameters"])


class FinetuneInitStrategy(InitStrategy):
    def initialize(self, model, parameters):
        path = parameters["estimator_settings"]["finetune_model_path"]
        # weights only false because seralizing lazyframe doesn't work
        # TODO fix
        fitted_estimator = torch.load(path, map_location="cpu", weights_only=False)
        fitted_parameters = fitted_estimator["model_parameters"]
        model_instance = model(**fitted_parameters)
        model_instance.load_state_dict(fitted_estimator["model_state_dict"])
        for param in model_instance.parameters():
            param.requires_grad = False
        model_instance.reset_head()
        return model_instance


class CustomEmbeddingInitStrategy(InitStrategy):
    def __init__(self, embedding_class: str, embedding_file: str, freeze: bool = True,
                 curvature: float = 1.0):
        self.embedding_class = embedding_class
        self.embedding_file = embedding_file
        self.freeze = freeze
        self.curvature = float(curvature)
        self.class_names_to_class = {
            "CustomEmbeddings": CustomEmbeddings,
            "PoincareEmbeddings": PoincareEmbeddings,
        }

    def initialize(self, model, parameters):
        file_path = pathlib.Path(self.embedding_file)
        data_reference = (
            parameters["model_parameters"]["feature_info"].data_reference
            .with_columns(pl.col("conceptId").cast(pl.Int64))
        )

        model = DefaultInitStrategy().initialize(model, parameters) 
        # files saved from GPU tensors load on machines without that GPU
        embeddings = torch.load(file_path, map_location="cpu", weights_only=True)

        if "concept_ids" not in embeddings.keys():
            raise KeyError(
                "The embeddings file does not contain the required 'concept_ids' key"
            )
        if "embeddings" not in embeddings.keys():
            raise KeyError(
                "The embeddings file does not contain the required 'embeddings' key"
            )
        if embeddings["concept_ids"].dtype != torch.long:
            raise TypeError(
                "The 'concept_ids' key in the embeddings file must be of type torch.long"
            )
        if embeddings["embeddings"].dtype != torch.float:
            raise TypeError(
                "The 'embeddings' key in the embeddings file must be of type torch.float"
            )

        # Ensure that the model has an embedding layer
        if not hasattr(model, "embedding"):
            raise AttributeError(
                f"The model: {model.name} does not have an embedding layer named 'embedding' as "
                f"required for custom embeddings"
            )

        # Match each covariate column to its row in the embeddings file. Columns
        # are one-based because zero is the padding index. Concepts in the file
        # that are not in the training data are out of vocabulary and dropped.
        embedding_rows = pl.DataFrame(
            {
                "conceptId": embeddings["concept_ids"].tolist(),
                "embeddingRow": list(range(embeddings["concept_ids"].shape[0])),
            },
            schema_overrides={"conceptId": pl.Int64},
        )
        matches = (
            data_reference.select("conceptId", "columnId")
            .join(embedding_rows, on="conceptId", how="inner")
            .sort("embeddingRow", "columnId")
        )
        # with several vectors for one concept there is no right one to choose
        repeated = (
            matches.group_by("conceptId")
            .agg(pl.col("embeddingRow").n_unique().alias("vectors"))
            .filter(pl.col("vectors") > 1)
            .sort("conceptId")
        )
        if repeated.height > 0:
            shown = ", ".join(str(concept) for concept in repeated["conceptId"].head(10).to_list())
            more = " and more" if repeated.height > 10 else ""
            raise ValueError(
                "The embeddings file has more than one vector for concept IDs "
                f"{shown}{more}. Keep one vector per concept."
            )
        custom_indices = torch.as_tensor(matches["columnId"].to_list(), dtype=torch.long)
        custom_weights = embeddings["embeddings"][matches["embeddingRow"].to_list()]
        if model.name == "ResNet" or model.name == "MultiLayerPerceptron":
            aggregate = "sum"
        else:
            aggregate = "none"

        embedding_class = self.class_names_to_class[self.embedding_class]
        extra = dict()
        if embedding_class is PoincareEmbeddings:
            # strategies saved with models from before curvature was a setting
            extra["curvature"] = getattr(self, "curvature", 1.0)
        model.embedding = embedding_class(
            custom_embedding_weights=custom_weights,
            embedding_dim=model.embedding.embedding_dim,
            feature_info=parameters["model_parameters"]["feature_info"],
            custom_indices=custom_indices,
            freeze=self.freeze,
            aggregate=aggregate,
            **extra,
        )
        return model
