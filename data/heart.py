from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.utils.data import DataLoader as dl
import numpy as np



def _get_np(center_id: int, train: bool, config: dict):
    center = FedHeartDisease(center=center_id, train=train)
    if config["split"] == True:
        c0_iter = iter(dl(center, batch_size=None, num_workers=0))
        X_train = np.empty((1,13))
        y_train = np.empty((1,))
        for point, target in c0_iter:
            small_x = point.numpy().reshape(1,13)
            small_y = target.numpy()
            X_train = np.append(X_train, small_x, axis=0)
            y_train = np.append(y_train, small_y, axis=0)
        X_train = X_train[1:]
        y_train = y_train[1:]
    else:
        X_train = center
        y_train = None
    return X_train, y_train


def get_client_data(center_id: int, config: dict):
    X_train, y_train = _get_np(center_id, train = True, config = config)
    print(len(X_train))
    X_val, y_val = _get_np(center_id, train = False, config = config)
    print(len(X_val))
    return X_train, y_train, X_val, y_val

def get_server_data(config: dict):
    return _get_np(1, train = False, config = config)

if __name__ == "__main__":
    config = {"split": True}
    X_train, y_train, X_val, y_val = get_client_data(0, config)
    X_test, y_test = get_server_data(config)