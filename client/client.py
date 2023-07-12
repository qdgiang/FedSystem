from typing import Any
import flwr as fl
import client.model.logistic_regression as model_chosen
from sklearn.linear_model import LogisticRegression

class MyClient(fl.client.NumPyClient):
    def __init__(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        device: str,
        model,
        model_name: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.validation_split = validation_split
        self.model = model
        self.model_name = model_name
        
    def __set_parameters(self, parameters):
        model_chosen.set_model_parameters(self.model, parameters)

    def get_parameters(self, config):
        return model_chosen.get_model_parameters(self.model)
    
    def __train(self, config):
        return model_chosen.fit(self.model, self.X_train, self.y_train, config)

    def __evaluate(self, config):
        return model_chosen.evaluate(self.model, self.X_eval, self.y_eval)
    
    def fit(self, parameters, config):
        self.__set_parameters(parameters)
        print(config)
        return self.__train(config)

    def evaluate(self, parameters, config):
        self.__set_parameters(parameters)
        print("Now evaluate")
        print(config)
        return self.__evaluate(config)

