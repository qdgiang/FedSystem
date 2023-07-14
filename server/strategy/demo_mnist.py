import flwr as fl
import yaml
import openml
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

# import ModelManager from model/model_manager.py
# from model.model_manager import ModelManager does not work
# use another!


import sys
sys.path.append("..") 
from model.model_manager import ModelManager
from data.data_manager import DataManager
from model.logistic_regression import set_parameters

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    temp_man = DataManager("mnist", "server")
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X_test = temp_man.get_test_data()
    y_test = temp_man.get_test_label()
    # use the first 10 samples of the test set
    randX = X_test[:100]
    randY = y_test[:100]
    model.fit(randX, randY)
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        set_parameters(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


model = LogisticRegression()
def fit_round(server_round: int):
    """Send round number to client."""
    return {"server_round": server_round}

base_mnist_strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round
    )