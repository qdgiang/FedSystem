from typing import Any
import sys
sys.path.append('.')
import flwr as fl
import model.LogisticRegression as model_chosen
import argparse
import torch
from data import MNIST
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
        # if self.model_name == 'LogisticRegression':
        # import fit function from model/LogisticRegression.py
        #if self.model_name == 'LogisticRegression':
        #model_chosen.train(self.model, self.X_train, self.y_train, epochs=1)
        return self.__train(config)

    def evaluate(self, parameters, config):
        self.__set_parameters(parameters)
        print(config)
        return self.__evaluate(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--partition_id",
        type=int,
        default=0,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    # Load a subset of CIFAR-10 to simulate the local data partition
    (X_train, y_train), (X_eval, y_eval) = MNIST.MNIST_DATA(args.partition_id).get_data()
    my_model = LogisticRegression()
    model_chosen.set_initial_params(my_model)
    # Start Flower client
    client = MyClient(X_train= X_train, 
                      y_train= y_train, 
                      X_eval= X_eval, 
                      y_eval= y_eval, 
                      device= device, 
                      model= my_model, 
                      model_name= "LogisticRegression")

    fl.client.start_numpy_client(server_address="127.0.0.1:6969", client=client)
if __name__ == "__main__":
    main()