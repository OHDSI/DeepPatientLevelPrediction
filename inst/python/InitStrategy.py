from abc import ABC, abstractmethod

import torch
import os
from torchviz import make_dot

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
        cat_2_features_default = 20  # Set a default value if you don't have one
        model_parameters['cat_2_features'] = model_parameters.get('cat_2_features', cat_2_features_default)

        # Instantiate the model with the provided parameters
        model_temp = model(**model_parameters)

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

        if file_path and os.path.exists(file_path):
            state = torch.load(file_path)
            state_dict = state["state_dict"]
            embedding_key = "embedding.weight"  # Key in the state dict for the embedding

            if embedding_key not in state_dict:
                raise KeyError(f"The key '{embedding_key}' does not exist in the state dictionary")

            new_embeddings = state_dict[embedding_key].float()

            # Ensure that model_temp.categorical_embedding_2 exists
            if not hasattr(model_temp, 'categorical_embedding_2'):
                raise AttributeError("The model does not have an attribute 'categorical_embedding_2'")

            # Replace the weights of `model_temp.categorical_embedding_2`
            if isinstance(model_temp.categorical_embedding_2, torch.nn.Embedding):
                with torch.no_grad():
                    model_temp.categorical_embedding_2.weight = torch.nn.Parameter(new_embeddings)
            else:
                raise TypeError("The attribute 'categorical_embedding_2' is not of type `torch.nn.Embedding`")

            print("New Embeddings:")
            print(new_embeddings)
            print(f"Restored Epoch: {state['epoch']}")
            print(f"Restored Mean Rank: {state['mean_rank']}")
            print(f"Restored Loss: {state['loss']}")
            print(f"Restored Names: {state['names']}")
        else:
            raise FileNotFoundError(f"File not found or path is incorrect: {file_path}")

        # Visualize the modified model architecture again
        try:
            output = model_temp(dummy_input)
            modified_graph = make_dot(output, params=dict(model_temp.named_parameters()))
            modified_graph.render("modified_model_architecture", format="png")
            print("Modified model architecture rendered successfully.")
        except Exception as e:
            print(f"Error during modified model visualization: {e}")

        return model_temp