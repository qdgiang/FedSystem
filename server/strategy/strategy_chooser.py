import importlib

def weighted_average(metrics): #List[Tuple[int, Metrics]]) -> Metrics:
    print("metrics: ", metrics)
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print("accuracies: ", accuracies)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def strategy_chooser(strategy_name: str):
    strategy_name = "." + strategy_name
    print("Hmm")
    print(strategy_name)
    strategy_file = importlib.import_module(strategy_name, package="strategy")
    if strategy_name == ".fedavg":
        print("Noice")
        return strategy_file.BaseFedAvg(
            fraction_fit=0.6,
            min_available_clients=4,
            min_fit_clients=3,
            fraction_evaluate=0.1,
            min_evaluate_clients=2,
            #fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            )
    else:
        return None
