import flwr as fl
from server_utils import fit_config, evaluate_config
import yaml
import openml
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


with open("server.yaml", "r") as f:
    server_config = yaml.safe_load(f)

print(type(server_config))


def load_mnist(): #-> Dataset:
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    # First 60000 samples consist of the train set
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)

def set_model_params(
    model, params
):
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = load_mnist()
    # use the first 10 samples of the test set
    X_test = X_test[:100]
    y_test = y_test[:100]
    model.fit(X_test, y_test)
    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        set_model_params(model, parameters)
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
        on_fit_config_fn=fit_round,
        #on_fit_config_fn=fit_round,
    )