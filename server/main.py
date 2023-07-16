from typing import Optional
import flwr as fl
from strategy.strategy_chooser import strategy_chooser
from dataclasses import dataclass
import yaml

@dataclass
class MyConfig:
    num_rounds: int = 1
    round_timeout: Optional[float] = None
    hello: str = "world"
    


def main():
    with open("../common.yaml", "r") as f:
        common_config = yaml.safe_load(f)

    with open("server.yaml", "r") as f:
        server_config = yaml.safe_load(f)

    strategy_name = "base_mnist_strategy"

    fl.server.start_server(
        server_address=server_config["server_address"],
        strategy=strategy_chooser(server_config["strategy"]),
        config=MyConfig(num_rounds=20)
    )


if __name__ == "__main__":
    main()
