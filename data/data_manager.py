import importlib

class DataManager:
    def __init__(self, 
                 data_name: str,
                 node_type: str = "client",
                 config: dict = None
    ) -> None:
        assert data_name in ["mnist", "cifar"]
        self.data_name = data_name
        self.node_type = node_type
        self.partition_id = config["partition_id"] if config is not None else 0
        data_source_name = "." + self.data_name
        data_source = importlib.import_module(data_source_name, package="data") # dynamically import the correct data module
        if node_type == "client":
            self.X_train, self.y_train, self.X_val, self.y_val = data_source.get_client_data(self.partition_id)
        else:
            (self.X_test, self.y_test) = data_source.get_server_data()

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
    data_manager = DataManager("mnist", "client", 0)
    #data_manager = DataManager("mnist", "server", 0)
    print("=====")
    data_manager2 = DataManager("mnist", "server")