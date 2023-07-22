from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import numpy as np
from sklearn.metrics import log_loss
import warnings
XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def init_model(model_config: dict):
    model = LogisticRegression(
        penalty=model_config.get("penalty", "l2"),
        dual=model_config.get("dual", False),
        tol=model_config.get("tol", 0.0001),
        C=model_config.get("C", 1.0),
        max_iter=model_config.get("max_iter", 1),
        warm_start=model_config.get("warm_start", True)
    )
    _set_initial_params(model, model_config)
    return model

def get_parameters(model: LogisticRegression) -> LogRegParams:
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
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def fit(
    model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray, config: dict
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    model.fit(X_train, y_train)
    return get_parameters(model), len(X_train), {}

def evaluate(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray, config: dict
):
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return loss, len(X_test), {"accuracy": accuracy}

######################################

def _set_initial_params(model: LogisticRegression, model_config: dict):
    print(model_config)
    n_classes = model_config.get("n_classes", 2)
    n_features = model_config.get("n_features", 13)
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))