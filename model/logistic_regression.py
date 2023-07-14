from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import numpy as np
from sklearn.metrics import log_loss

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def init_model(config: dict):
    """Initializes a sklearn LogisticRegression model."""

    model = LogisticRegression(
        penalty=config.get("penalty", "l2"),
        dual=config.get("dual", False),
        tol=config.get("tol", 0.0001),
        C=config.get("C", 1.0),
        max_iter=config.get("max_iter", 100),
        warm_start=config.get("warm_start", False)
    )
    _set_initial_params(model, config)
    return model

def get_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_parameters(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def fit(
    model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray, config: dict = None
) -> LogisticRegression:
    """Trains a sklearn LogisticRegression model."""
    model.fit(X_train, y_train)
    return get_parameters(model), len(X_train), {}

def evaluate(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray, config: dict = None
) -> float:
    """Evaluates a sklearn LogisticRegression model."""
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return loss, len(X_test), {"accuracy": accuracy}

######################################

def _set_initial_params(model: LogisticRegression, config: dict):
    """Sets initial parameters as zeros. 
    Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = config.get("n_classes", 10)
    n_features = config.get("n_features", 784)
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))