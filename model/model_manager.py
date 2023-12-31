import importlib
import yaml
import os
from common.logger import FED_LOGGER

class ModelManager:
    def __init__(self, model_name: str) -> None:
        assert model_name in ["logistic", "cnn", "svm", "mlp"]
        config_location = f"{os.path.dirname(__file__)}/config/{model_name}.yaml"
        with open(config_location, "r") as f:
            self.model_config = yaml.safe_load(f)
        self.model_name = model_name
        self.model_source = importlib.import_module(name=f"model.{self.model_name}")
        self.model = self.model_source.init_model(self.model_config)
        FED_LOGGER.info(f"Model initialized successfully! Model type: {type(self.model)}")
    
    def get_params(self):
        return self.model_source.get_parameters(self.model)
    
    def set_params(self, parameters):
        self.model_source.set_parameters(self.model, parameters)

    def get_model(self):
        return self.model
    
    def fit_model(self, X_train, y_train, additonal_config):
        return self.model_source.fit(self.model, X_train, y_train, self.model_config | additonal_config)
    
    def evaluate_model(self, X_test, y_test): # also use for server model final central testing
        return self.model_source.evaluate(self.model, X_test, y_test, self.model_config)

if __name__ == "__main__":
    model_manager = ModelManager("cnn")
