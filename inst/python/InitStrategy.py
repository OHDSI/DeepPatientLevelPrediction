from abc import ABC, abstractmethod
import torch

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

