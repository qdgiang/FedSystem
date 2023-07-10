from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import numpy as np
from sklearn.metrics import log_loss

XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def get_model_parameters(model: LogisticRegression) -> LogRegParams:
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

def set_model_parameters(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def fit(
    model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray, config
) -> LogisticRegression:
    """Trains a sklearn LogisticRegression model."""
    model.fit(X_train, y_train)
    return get_model_parameters(model), len(X_train), {}

def evaluate(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluates a sklearn LogisticRegression model."""
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)
    return loss, len(X_test), {"accuracy": accuracy}

def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))