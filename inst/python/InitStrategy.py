from abc import ABC, abstractmethod
import pathlib

import torch
import polars as pl

from CustomEmbeddings import CustomEmbeddings, PoincareEmbeddings


class InitStrategy(ABC):
    @abstractmethod
    def initialize(self, model, parameters):
        pass


class DefaultInitStrategy(InitStrategy):
    def initialize(self, model, parameters):
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
    def __init__(self, embedding_class: str, embedding_file: str, freeze: bool = True):
        self.embedding_class = embedding_class
        self.embedding_file = embedding_file
        self.freeze = freeze
        self.class_names_to_class = {
            "CustomEmbeddings": CustomEmbeddings,
            "PoincareEmbeddings": PoincareEmbeddings,
        }

    def initialize(self, model, parameters):
        file_path = pathlib.Path(self.embedding_file)
        data_reference = parameters["model_parameters"]["feature_info"].data_reference

        # Instantiate the model with the provided parameters
        model = model(**parameters["model_parameters"])

        embeddings = torch.load(file_path, weights_only=True)

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

        # get indices of the custom embeddings from embeddings["concept_ids"]
        # I need to select the rows from data_reference where embeddings["concept_ids"] is in data_reference["conceptId"]
        # data reference is a polars lazyframe. Note at this point data_reference only contains concepts in the training set
        custom_indices = (
            data_reference.filter(
                pl.col("conceptId").is_in(embeddings["concept_ids"].tolist())
            ).select("columnId")
            - 1
        )
        custom_indices = custom_indices.to_torch().squeeze()
        # filter embeddings to concepts in training data, rest is OOV (out of vocabulary)
        concepts_in_data = data_reference.select("conceptId")
        embeddings = {
            k: v[
                torch.isin(
                    embeddings["concept_ids"], concepts_in_data["conceptId"].to_torch()
                )
            ]
            for k, v in embeddings.items()
        }
        if model.name == "ResNet" or model.name == "MultiLayerPerceptron":
            aggregate = "sum"
        else:
            aggregate = "none"

        embedding_class = self.class_names_to_class[self.embedding_class]
        model.embedding = embedding_class(
            custom_embedding_weights=embeddings["embeddings"],
            embedding_dim=model.embedding.embedding_dim,
            feature_info=parameters["model_parameters"]["feature_info"],
            custom_indices=custom_indices,
            freeze=self.freeze,
            aggregate=aggregate
        )
        return model
