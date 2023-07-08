import flwr as fl
from strategy import base_strategy
import yaml

def main():

    #with open("server.yaml", "r") as f:
    #    server_config = yaml.safe_load(f)


    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        #config=fl.server.ServerConfig(num_rounds=10),
        strategy=base_strategy,
    )


if __name__ == "__main__":
    main()
