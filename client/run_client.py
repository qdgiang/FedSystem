from typing import Any
import flwr as fl
import client.model.logistic_regression as model_chosen
import argparse
import torch
from sklearn.linear_model import LogisticRegression
from data.mnist import MnistDataClass
from .client import MyClient

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
    (X_train, y_train), (X_eval, y_eval) = MnistDataClass(args.partition_id).get_data()
    my_model = LogisticRegression(
        #penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

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