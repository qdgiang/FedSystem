import openml
import numpy as np

def load_mnist():  # -> Dataset
    """Loads the MNIST dataset using OpenML.

    OpenML dataset link: https://www.openml.org/d/554
    """
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(
        target=None,
        include_row_id=False,
        include_ignore_attribute=False,
        dataset_format="array"
        )
    X = Xy[:, :-1]  # the last column contains labels
    y = Xy[:, -1]
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int): # -> XYList:
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )

def get_client_data(partition_id: int, config: dict = None):
    (full_data, full_label), (_, _) = load_mnist()
    (client_data, client_label) = partition(full_data,full_label, 10)[partition_id]
    X_train, y_train = client_data[:5000], client_label[:5000]
    X_val, y_val = client_data[5000:], client_label[5000:]
    return X_train, y_train, X_val, y_val


def get_server_data(config: dict = None):
    (_, _), (X_test, y_test) = load_mnist()
    return X_test, y_test