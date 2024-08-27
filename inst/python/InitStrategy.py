from abc import ABC, abstractmethod

import torch
import os
from torchviz import make_dot
import json
import polars as pl

class InitStrategy(ABC):
    @abstractmethod
    def initialize(self, model, model_parameters, estimator_settings):
        pass
      
class DefaultInitStrategy(InitStrategy):
    def initialize(self, model, model_parameters, estimator_settings):
        return model(**model_parameters)

class FinetuneInitStrategy(InitStrategy):
    def initialize(self, model, model_parameters, estimator_settings):
        path = estimator_settings["finetune_model_path"]
        fitted_estimator = torch.load(path, map_location="cpu")
        fitted_parameters = fitted_estimator["model_parameters"]
        model_instance = model(**fitted_parameters)
        model_instance.load_state_dict(fitted_estimator["model_state_dict"])
        for param in model_instance.parameters():
            param.requires_grad = False
        model_instance.reset_head()
        return model_instance


class CustomEmbeddingInitStrategy(InitStrategy):
    def initialize(self, model, model_parameters, estimator_settings):
        file_path = estimator_settings.get("embedding_file_path")

        # Ensure `cat_2_features` is added to `model_parameters`
        # cat_2_features_default = 20  # Set a default value if you don't have one
        print(model_parameters['cat_2_features'])
        print(model_parameters['cat_features'])
        print(model_parameters['num_features'])


        # Instantiate the model with the provided parameters
        model_temp = model(**model_parameters)

        if file_path and os.path.exists(file_path):
            state = torch.load(file_path)
            state_dict = state["state_dict"]
            embedding_key = "embedding.weight"

            if embedding_key not in state_dict:
                raise KeyError(f"The key '{embedding_key}' does not exist in the state dictionary")

            new_embeddings = state_dict[embedding_key].float()
            print(f"new_embeddings: {new_embeddings}")

            # Ensure that model_temp.categorical_embedding_2 exists
            if not hasattr(model_temp, 'categorical_embedding_2'):
                raise AttributeError("The model does not have an attribute 'categorical_embedding_2'")

            # # replace weights
            # cat2_concept_mapping = pl.read_json(os.path.expanduser("~/Desktop/cat2_concept_mapping.json"))
            cat2_mapping = pl.read_json(os.path.expanduser("~/Desktop/cat2_mapping.json"))
            print(f"cat2_mapping: {cat2_mapping}")

            concept_df = pl.DataFrame({"conceptId": state['names']}).with_columns(pl.col("conceptId"))
            print(f"concept_df: {concept_df}")

            # Initialize tensor for mapped embeddings
            mapped_embeddings = torch.zeros((cat2_mapping.shape[0] + 1, new_embeddings.shape[1]))
            
            # Map embeddings to their corresponding indices
            for row in cat2_mapping.iter_rows():
                concept_id, covariate_id, index = row
                if concept_id in concept_df["conceptId"]:
                    concept_idx = concept_df["conceptId"].to_list().index(concept_id)
                    mapped_embeddings[index] = new_embeddings[concept_idx]
            
            print(f"mapped_embeddings: {mapped_embeddings}")
            
            # Assign the mapped embeddings to the model
            model_temp.categorical_embedding_2.weight = torch.nn.Parameter(mapped_embeddings)
            model_temp.categorical_embedding_2.weight.requires_grad = False
            
            print("New Embeddings:")
            print(new_embeddings)
            print(f"Restored Epoch: {state['epoch']}")
            print(f"Restored Mean Rank: {state['mean_rank']}")
            print(f"Restored Loss: {state['loss']}")
            print(f"Restored Names: {state['names'][:5]}")
            print(f"Number of names: {len(state['names'])}")
            # print(f"Filtered Embeddings: {filtered_embeddings}")
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

        if hasattr(model_temp, 'forward'):
            try:
                output = model_temp(dummy_input)
                initial_graph = make_dot(output, params=dict(model_temp.named_parameters()), show_attrs=False, show_saved=False)
                initial_graph.render("initial_model_architecture", format="png")
            except Exception as e:
                print(f"Error during initial model visualization: {e}")

        else:
            raise AttributeError("The model does not have a forward method.")
        
        return model_temp
