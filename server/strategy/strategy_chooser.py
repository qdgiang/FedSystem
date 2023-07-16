import importlib

def strategy_chooser(strategy_name: str):
    strategy_name = "." + strategy_name
    strategy_file = importlib.import_module(strategy_name, package="strategy")
    if strategy_name == "fedavg":
        return strategy_file.BaseFedAvg()
    else:
        return None
