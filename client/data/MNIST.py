import openml
import numpy as np
from base_data import BaseDataClass

class MNIST_DATA(BaseDataClass):
    def __load_mnist(self):  # -> Dataset
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

    def __partition(self, X: np.ndarray, y: np.ndarray, num_partitions: int): # -> XYList:
        """Split X and y into a number of partitions."""
        return list(
            zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
        )

    def get_data(self):
        (X_train, y_train), (X_test, y_test) = self.__load_mnist()
        (X_train, y_train) = self.__partition(X_train, y_train, 10)[self.partition_id]
        return (X_train, y_train), (X_test, y_test)