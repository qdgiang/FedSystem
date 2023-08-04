def init_model(model_config: dict):
    """
    Create a model instance based on the model_config.
    """
    pass

def get_parameters(model):
    """
    Return the learned parameters of the model.
    """
    pass

def set_parameters(model, parameters):
    """
    Set the parameters of the model to the given parameters.
    """
    pass

def fit(model, X_train, y_train, train_config: dict):
    """
    Fit the model on the local training data.
    Use the train_config to define how we want to train the model (epochs, verbose, etc.)
    """
    pass

def evaluate(model, X_test, y_test):
    """
    Evaluate the model on the local testing data.
    Optional: This can also be used by server for final testing metrics on a server test dataset if available.
    """
    pass
