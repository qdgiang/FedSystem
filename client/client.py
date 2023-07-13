import flwr as fl

from data.data_manager import DataManager
from model.model_manager import ModelManager

class MyClient(fl.client.NumPyClient):
    def __init__(
        self,
        data_manager: DataManager,
        model_manager: ModelManager,
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
     
        
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

