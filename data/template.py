def get_client_data(partition_id: int, config: dict):
    """
    Return 4 values X_train, y_train, X_val, y_val
    Corresponding to features and labels for the training and evaluating dataset.
    Use config to define how we want to federate the data if it is not already so.
    """
    pass

def get_server_data():
    """
    Return 2 values X_test, y_test
    Corresponding to features and labels for the final centralized testing dataset.
    This is optional, if we want to have a final testing dataset after finishing federated training. 
    """
    pass