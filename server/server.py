import flwr as fl
#from strategy import base_strategy, mnist_strategy
import yaml

def main():

    #with open("server.yaml", "r") as f:
    #    server_config = yaml.safe_load(f)

    strategy_name = "base_mnist_strategy"
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:6969",
        #config=fl.server.ServerConfig(num_rounds=10),
        strategy=mnist_strategy.base_mnist_strategy,
        config=fl.server.ServerConfig(num_rounds=5)
    )


if __name__ == "__main__":
    main()
