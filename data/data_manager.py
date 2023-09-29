import importlib
from common.logger import FED_LOGGER
import torch

class DataManager:
    def __init__(self, node_type: str, config: dict) -> None:
        data_name = config["data"]
        assert data_name in ["mnist", "cifar", "heart", "tcga", "advanced"]
        self.data_name = data_name
        self.node_type = node_type
        self.data_config = config
        self.cid = config["cid"] #if config is not None else 0

        data_source = importlib.import_module(f"data.{self.data_name}")
        if node_type == "client":
            self.X_train, self.y_train, self.X_val, self.y_val = data_source.get_client_data(self.cid, self.data_config)
        else:
            (self.X_test, self.y_test) = data_source.get_server_data(self.data_config)
        FED_LOGGER.info(f"[{self.node_type} {self.cid}] Data imported sucessfully. Data name: {self.data_name}")

    def randomized_label(self):
        FED_LOGGER.info(f"[{self.node_type} {self.cid}] Randomized label")
        self.X_train.dataset.dataset.dataset.targets = torch.randint(0, 10, (50000,))

    def set_all_label(self, label):
        FED_LOGGER.info(f"[{self.node_type} {self.cid}] Set all label to {label}")
        self.X_train.dataset.dataset.dataset.targets = torch.ones(50000, dtype=torch.int64) * label

    def set_original_label(self):
        FED_LOGGER.info(f"[{self.node_type} {self.cid}] Set original label")
        data_source = importlib.import_module(f"data.{self.data_name}")
        if self.node_type == "client":
            self.X_train, self.y_train, self.X_val, self.y_val = data_source.get_client_data(self.cid, self.data_config)
        else:
            (self.X_test, self.y_test) = data_source.get_server_data(self.data_config)
        FED_LOGGER.info(f"[{self.node_type} {self.cid}] Data imported sucessfully. Data name: {self.data_name}")

    def get_training_data(self):
        return self.X_train
    
    def get_training_label(self):
        return self.y_train
    
    def get_eval_data(self):
        return self.X_val
    
    def get_eval_label(self):
        return self.y_val
    
    def get_test_data(self):
        return self.X_test
    
    def get_test_label(self):
        return self.y_test
    
if __name__ == "__main__":
    data_manager = DataManager("client", {"data": "heart", "cid": 0, "split": True})