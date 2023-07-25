import yaml
from strategy import strategy_chooser
from utils import start_server, MyServerConfig

def main():
    with open(file="../config.yaml", mode="r") as f:
        server_config = yaml.safe_load(f).get("server")
    print("Server_config: ", server_config)
    start_server(
        server_address=server_config["server_address"],
        strategy=strategy_chooser(server_config["strategy"], {}),
        config=MyServerConfig(num_rounds=30)
    )

if __name__ == "__main__":
    main()
