from abc import ABC, abstractmethod
import pathlib

import torch
import polars as pl

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
        file_path = pathlib.Path(parameters["estimator_settings"].get("embedding_file_path")).expanduser()

        # Instantiate the model with the provided parameters
        model = model(**parameters["model_parameters"])

        if file_path.exists():
            state = torch.load(file_path)
            state_dict = state["state_dict"]
            embedding_key = "embedding.weight"

            if embedding_key not in state_dict:
                raise KeyError(f"The key '{embedding_key}' does not exist in the state dictionary")

            custom_embeddings = state_dict[embedding_key].float()

            # Ensure that model_temp.categorical_embedding_2 exists
            if not hasattr(model, 'embedding'):
                raise AttributeError("The model does not have an embedding layer named 'embedding'")

            # # replace weights
            cat2_mapping = pl.read_json(os.path.expanduser("~/Desktop/cat2_mapping_train.json"))

            concept_df = pl.DataFrame({"conceptId": state['names']}).with_columns(pl.col("conceptId"))
            
            # Initialize tensor for mapped embeddings
            mapped_embeddings = torch.zeros((cat2_mapping.shape[0] + 1, new_embeddings.shape[1]))
            
            # Map embeddings to their corresponding indices
            for row in cat2_mapping.iter_rows():
                concept_id, covariate_id, index = row
                if concept_id in concept_df["conceptId"]:
                    concept_idx = concept_df["conceptId"].to_list().index(concept_id)
                    mapped_embeddings[index] = new_embeddings[concept_idx]
            
            # Assign the mapped embeddings to the model
            model_temp.categorical_embedding_2.weight = torch.nn.Parameter(mapped_embeddings)
            model_temp.categorical_embedding_2.weight.requires_grad = False
            
        else:
            raise FileNotFoundError(f"File not found or path is incorrect: {file_path}")


        # Create a dummy input batch that matches the model inputs
        dummy_input = {
            "cat": torch.randint(0, model_parameters['cat_features'], (1, 10)).long(),
            "cat_2": torch.randint(0, model_parameters['cat_2_features'], (1, 10)).long(),
            "num": torch.randn(1, model_parameters['num_features']) if model_parameters['num_features'] > 0 else None
        }

        # Ensure that the dummy input does not contain `None` values if num_features == 0
        if model_parameters['num_features'] == 0:
            del dummy_input["num"]

        return model_temp
