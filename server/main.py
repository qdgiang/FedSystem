from typing import Optional
from strategy.strategy_chooser import strategy_chooser
from dataclasses import dataclass
import yaml
from utils.app import ServerConfig, start_server

@dataclass
class MyConfig:
    num_rounds: int = 1
    round_timeout: Optional[float] = None
    hello: str = "world"
    


def main():
    with open(file="../config.yaml", mode="r") as f:
        server_config = yaml.safe_load(f).get("server")
    print("*" * 20)
    print("Server_config: ", server_config)
    start_server(
        server_address=server_config["server_address"],
        strategy=strategy_chooser(server_config["strategy"], {}),
        config=ServerConfig(num_rounds=30)
    )

if __name__ == "__main__":
    main()
