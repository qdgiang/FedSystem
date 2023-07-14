import flwr as fl
import argparse
import torch

from .client import MyClient
from data.data_manager import DataManager
from model.model_manager import ModelManager

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
        required=True,
        help="Set to true to use GPU. Default: False",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    print("Using device: %s" % device)

    data_config = {
        "partition_id": args.partition_id,
    }

    model_config = {
        "device": device, 
        "n_classes": 10, 
        "n_features": 784
    }

    data_manager = DataManager("mnist", "client", data_config)
    model_manager = ModelManager("logistic_regression", model_config)
    client = MyClient(data_manager, model_manager)
    fl.client.start_numpy_client(server_address="127.0.0.1:6969", client=client)
    
if __name__ == "__main__":
    main()