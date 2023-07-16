import flwr as fl
import argparse
import torch
import yaml
from client import MyClient
from data.data_manager import DataManager
from model.model_manager import ModelManager

def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition_id",
        type=int,
        default=0,
        required=True,
        help="Set to true to use GPU. Default: False",
    )
    args = parser.parse_args()

    with open("../common.yaml", "r") as f:
        common_config = yaml.safe_load(f)

    with open("client.yaml", "r") as f:
        client_config = yaml.safe_load(f)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and common_config["cuda"] == True else "cpu"
    )

    data_config = {
        "partition_id": args.partition_id,
    }

    model_config = {
        "device": device, 
        "n_classes": 10, 
        "n_features": 784
    }

    data_manager = DataManager(
        common_config["data"], 
        "client", 
        data_config
    )

    model_manager = ModelManager(
        common_config["model"], 
        model_config
    )

    client = MyClient(data_manager, model_manager)
    fl.client.start_numpy_client(server_address=client_config["server_address"], client=client)
    
if __name__ == "__main__":
    main()