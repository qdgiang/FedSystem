import importlib

class DataManager:
    def __init__(self, 
                 data_name: str,
                 node_type: str = "client",
                 partition_id: int = 0,
    ) -> None:
        assert data_name in ["mnist", "cifar10", "cifar100"]
        self.data_name = data_name
        self.node_type = node_type
        self.partition_id = partition_id
        data_source = importlib.import_module(self.data_name) # dynamically import the correct data module
        (self.X_train, self.y_train), (self.X_val, self.y_val) = data_source.get_client_data(self.partition_id)
        print("X_train shape: ", self.X_train.shape)
        print("y_train shape: ", self.y_train.shape)
        print("X_val shape: ", self.X_val.shape)
        print("y_val shape: ", self.y_val.shape)


if __name__ == "__main__":
    data_manager = DataManager("mnist", "client", 0)
    #data_manager = DataManager("mnist", "server", 0)