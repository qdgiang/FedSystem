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

def set_model_params(
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
    return model

def evaluate(
    model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Evaluates a sklearn LogisticRegression model."""
    loss = log_loss(y_test, model.predict_proba(X_test))
    accuracy = model.score(X_test, y_test)
    return loss, len(X_test), {"accuracy": accuracy}
