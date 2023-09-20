import importlib
import os
import sys
sys.path.append((os.path.dirname(__file__)))
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(__file__)))))
from common.logger import FED_LOGGER

from flwr.server.strategy import Strategy
def weighted_average(metrics): #List[Tuple[int, Metrics]]) -> Metrics:
    FED_LOGGER.info("metrics: %s", metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print("accuracies: ", accuracies)
    return {"accuracy": sum(accuracies) / sum(examples)}

def strategy_chooser(strategy_name: str, strategy_config: dict) -> Strategy:
    strategy_file = importlib.import_module(strategy_name)
    if strategy_name == "fedavg":
        print("Strategy: FedAvg")
        return strategy_file.MyFedAvg(
            fraction_fit=1,
            min_available_clients=5,
            min_fit_clients=5,
            fraction_evaluate=0,
            min_evaluate_clients=0,
            #fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            )
    else:
        print("Strategy not found")
        return None

if __name__ == "__main__":
    strategy_chooser("fedavg", {})