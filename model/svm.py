from sklearn.svm import LinearSVC
from typing import Tuple, Union, List
import numpy as np
from sklearn.metrics import log_loss
import warnings
XY = Tuple[np.ndarray, np.ndarray]
LogRegParams = Union[XY, Tuple[np.ndarray]]

def init_model(model_config: dict):
    model = LinearSVC(
        #penalty=model_config.get("penalty", "l2"),
        #dual=model_config.get("dual", False),
        #tol=model_config.get("tol", 0.0001),
        #C=model_config.get("C", 1.0),
        max_iter=model_config.get("max_iter", 20),
        #warm_start=model_config.get("warm_start", True)
    )
    _set_initial_params(model, model_config)
    return model

def get_parameters(model: LinearSVC):
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
    model: LinearSVC, params: LogRegParams
):
    model.coef_ = params[0]
    #if model.fit_intercept:
    model.intercept_ = params[1]
    return model

def fit(
    model: LinearSVC, X_train: np.ndarray, y_train: np.ndarray, config: dict
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    model.fit(X_train, y_train)
    return get_parameters(model), len(X_train), {}

def evaluate(
    model: LinearSVC, X_test: np.ndarray, y_test: np.ndarray, config: dict
) -> float:
    loss = 0
    accuracy = 0

    for i in range(len(X_test)):
        x = X_test[i]
        y = y_test[i]
        y_pred = model.predict([x])[0]
        loss += max(0, 1 - y * y_pred)
        if y_pred == y:
            accuracy += 1
    loss /= len(X_test)
    accuracy /= len(X_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")
    return loss, len(X_test), {"accuracy": accuracy}

######################################

def _set_initial_params(model: LinearSVC, model_config: dict):
    n_classes = model_config.get("n_classes", 2)
    n_features = model_config.get("n_features", 13)

    model.classes_ = np.array([i for i in range(n_classes)])
    model.coef_ = np.zeros((n_classes, n_features))

    if model.fit_intercept is True:
        model.intercept_ = np.zeros((n_classes,))