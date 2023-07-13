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
        
        data_source = importlib.import_module(self.data_name) # dynamically import the correct data module
        if node_type == "client":
            self.X_train, self.y_train, self.X_val, self.y_val = data_source.get_client_data(self.partition_id)
            print("X_train shape: ", self.X_train.shape)
            print("y_train shape: ", self.y_train.shape)
            print("X_val shape: ", self.X_val.shape)
            print("y_val shape: ", self.y_val.shape)
        else:
            (self.X_test, self.y_test) = data_source.get_server_data()
            print("X_test shape: ", self.X_test.shape)
            print("y_test shape: ", self.y_test.shape)

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