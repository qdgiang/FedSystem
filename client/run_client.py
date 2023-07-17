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
        "--cid",
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
        "cid": args.cid,
    }

    data_manager = DataManager(
        common_config["data"], 
        "client", 
        data_config
    )

    model_manager = ModelManager(
        common_config["model"], 
    )

    client = MyClient(data_manager, model_manager)
    if client_config["test"] == False:
        fl.client.start_numpy_client(server_address=client_config["server_address"], client=client)
    
if __name__ == "__main__":
    main()