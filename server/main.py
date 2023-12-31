import yaml
import os

from strategy import strategy_chooser
from utils import start_server, MyServerConfig, MyServer, MyClientManager
from common.logger import FED_LOGGER
from data import DataManager
from model import ModelManager

def main():

    dir = os.path.dirname(os.path.dirname(__file__))
    
    with open(file=f"{dir}/config.yaml", mode="r") as f:
        all_config = yaml.safe_load(f)

    with open(file=f"{dir}/config.yaml", mode="r") as f:
        yaml_server_config = yaml.safe_load(f).get("server")
    with open(file=f"{dir}/config.yaml", mode="r") as f:
        common_config = yaml.safe_load(f).get("common")
    data_config = {
        "cid": -1,
        "data": common_config["data"],
        "advanced_name": common_config["advanced_name"],
        "split": common_config["split"],
    }
    data_manager = DataManager(
        "server", 
        data_config
    )
    model_manager = ModelManager(
        common_config["model"], 
    )

        
    FED_LOGGER.info("Initial all_config:\n%s", all_config)
    FED_LOGGER.info("Initial server_config:\n%s", yaml_server_config)
    

    start_server(
        server_address=yaml_server_config["server_address"],
        server=MyServer(
            client_manager=MyClientManager(),
            strategy=strategy_chooser(yaml_server_config["strategy"], {}),
            data_manager=data_manager,
            model_manager=model_manager,
        ),
        #strategy=strategy_chooser(yaml_server_config["strategy"], {}),
        config=MyServerConfig(num_rounds=yaml_server_config["num_rounds"])
    )

if __name__ == "__main__":
    main()
