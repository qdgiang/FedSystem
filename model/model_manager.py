import importlib

class ModelManager:
    def __init__(self, 
                 model_name: str,
                 config: dict,
    ) -> None:
        assert model_name in ["logistic_regression", "cnn"]
        self.model_name = model_name
        self.config = config
        self.model_source = importlib.import_module(model_name)
        self.model = self.model_source.init_model(config)

    def get_model(self):
        return self.model
    
    def get_params(self):
        return self.model_source.get_model_parameters(self.model)
    
    def set_params(self, parameters):
        self.model_source.set_model_parameters(self.model, parameters)
    
    def fit_model(self, X_train, y_train, config):
        return self.model_source.fit(self.model, X_train, y_train, config)

if __name__ == "__main__":
    model_manager = ModelManager("logistic_regression", {"n_classes": 10, "n_features": 784})
    print(model_manager.get_params())
