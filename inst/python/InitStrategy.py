from abc import ABC, abstractmethod
import pathlib

import torch
import polars as pl

from CustomEmbeddings import CustomEmbeddings

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
        fitted_estimator = torch.load(path, map_location="cpu")
        fitted_parameters = fitted_estimator["model_parameters"]
        model_instance = model(**fitted_parameters)
        model_instance.load_state_dict(fitted_estimator["model_state_dict"])
        for param in model_instance.parameters():
            param.requires_grad = False
        model_instance.reset_head()
        return model_instance


class CustomEmbeddingInitStrategy(InitStrategy):
    def initialize(self, model, parameters):
        file_path = pathlib.Path(parameters["estimator_settings"].get("embedding_file_path"))
        data_reference = parameters["model_parameters"]["feature_info"]["reference"]

        # Instantiate the model with the provided parameters
        model = model(**parameters["model_parameters"])

        embeddings = torch.load(file_path, weights_only=True)

        if "concept_ids" not in embeddings.keys() :
            raise KeyError(f"The embeddings file does not contain the required 'concept_ids' key")
        if "embeddings" not in embeddings.keys():
            raise KeyError(f"The embeddings file does not contain the required 'embeddings' key")
        if embeddings["concept_ids"].dtype != torch.long:
            raise TypeError(f"The 'concept_ids' key in the embeddings file must be of type torch.long")
        if embeddings["embeddings"].dtype != torch.float:
            raise TypeError(f"The 'embeddings' key in the embeddings file must be of type torch.float")

        # Ensure that the model has an embedding layer
        if not hasattr(model, 'embedding'):
            raise AttributeError(f"The model: {model.name} does not have an embedding layer named 'embedding' as "
                                 f"required for custom embeddings")

        # get indices of the custom embeddings from embeddings["concept_ids"]
        # I need to select the rows from data_reference where embeddings["concept_ids"] is in data_reference["conceptId"]
        # data reference is a polars lazyframe
        custom_indices = data_reference.filter(pl.col("conceptId").is_in(embeddings["concept_ids"].tolist())).select("columnId").collect() - 1
        custom_indices = custom_indices.to_torch().squeeze()

        model.embedding = CustomEmbeddings(custom_embedding_weights=embeddings["embeddings"],
                                           embedding_dim=model.embedding.embedding_dim,
                                           num_regular_embeddings=model.embedding.num_embeddings,
                                           custom_indices=custom_indices)
        return model
