import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import DataManager
from model import ModelManager
from flwr.client import NumPyClient
from common.logger import FED_LOGGER
class MyClient(NumPyClient):
    def __init__(
        self,
        data_manager: DataManager,
        model_manager: ModelManager,
    ):
        self.data_manager = data_manager
        self.model_manager = model_manager
        FED_LOGGER.info(f"Client {self.data_manager.cid} initialized")
        
    def set_parameters(self, parameters, config: dict):
        self.model_manager.set_params(parameters)

    def get_parameters(self, config: dict):
        return self.model_manager.get_params()
    
    def fit(self, parameters, config: dict):
        self.set_parameters(parameters, config)
        FED_LOGGER.info(f"[Client {self.data_manager.cid}] Local model fitting started!")
        updates, length, metrics = self.model_manager.fit_model(self.data_manager.get_training_data(), self.data_manager.get_training_label())
        FED_LOGGER.info(f"[Client {self.data_manager.cid}] Local model fitting finished! Metrics: {metrics}")
        return updates, length, metrics
    def evaluate(self, parameters, config: dict):
        self.set_parameters(parameters, config)
        FED_LOGGER.info(f"[Client {self.data_manager.cid}] Local model evaluation started!")
        return self.model_manager.evaluate_model(self.data_manager.get_eval_data(), self.data_manager.get_eval_label())
        

