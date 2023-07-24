import sys
sys.path.append("..")
from data.data_manager import DataManager
from model.model_manager import ModelManager
from utils.numpy_client import NumPyClient
class MyClient(NumPyClient):
    def __init__(
        self,
        data_manager: DataManager,
        model_manager: ModelManager,
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
     
        
    def set_parameters(self, parameters, config: dict):
        self.model_manager.set_params(parameters)

    def get_parameters(self, config: dict):
        return self.model_manager.get_params()
    
    def fit(self, parameters, config: dict):
        self.set_parameters(parameters, config)
        return self.model_manager.fit_model(self.data_manager.get_training_data(), self.data_manager.get_training_label())

    def evaluate(self, parameters, config: dict):
        self.set_parameters(parameters, config)
        return self.model_manager.evaluate_model(self.data_manager.get_eval_data(), self.data_manager.get_eval_label())
        
