import yaml
import os

from strategy import strategy_chooser
from utils import start_server, MyServerConfig

def main():

    dir = os.path.dirname(os.path.dirname(__file__))
    
    with open(file=f"{dir}/config.yaml", mode="r") as f:
        yaml_server_config = yaml.safe_load(f).get("server")
    print("Initial server_config: ", yaml_server_config)

    start_server(
        server_address=yaml_server_config["server_address"],
        strategy=strategy_chooser(yaml_server_config["strategy"], {}),
        config=MyServerConfig(num_rounds=yaml_server_config["num_rounds"])
    )

if __name__ == "__main__":
    main()
