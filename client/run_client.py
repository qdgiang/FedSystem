import argparse
import yaml
import os

from base_client import MyClient
from data import DataManager
from model import ModelManager
from utils.app import start_numpy_client

def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--cid",
        type=int,
        default=0,
        required=True,
        help="Specify the client ID",
    )
    args = parser.parse_args()
    
    dir = os.path.dirname(os.path.dirname(__file__))
    with open(f"{dir}/config.yaml", "r") as f:
        client_config = yaml.safe_load(f).get("client")
    with open(f"{dir}/config.yaml", "r") as f:
        common_config = yaml.safe_load(f).get("common")
    data_config = {
        "cid": args.cid,
        "data": common_config["data"],
        "advanced_name": common_config["advanced_name"],
        "split": common_config["split"],
    }

    data_manager = DataManager(
        "client", 
        data_config
    )
    model_manager = ModelManager(
        common_config["model"], 
    )
    
    client = MyClient(data_manager, model_manager)
    if not client_config["test"]:
        start_numpy_client(server_address=client_config["server_address"], client=client)
    
if __name__ == "__main__":
    main()