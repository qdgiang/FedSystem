import flwr as fl
from server_utils import fit_config, evaluate_config
import yaml

with open("server.yaml", "r") as f:
    server_config = yaml.safe_load(f)

print(type(server_config))

base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=server_config["server"]["fraction_fit"],
        fraction_evaluate=server_config["server"]["fraction_evaluate"],
        #min_fit_clients=server_config["server"]["min_fit_clients"],
        #min_evaluate_clients=server_config["min_evaluate_clients"],
        #min_available_clients=server_config["min_available_clients"],
        #evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )