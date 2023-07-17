from .FLamby.flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.utils.data import DataLoader as dl
import numpy as np



def _get_np(center_id: int, train: bool):
    center = FedHeartDisease(center=center_id, train=train)
    c0_iter = iter(dl(center, batch_size=None, num_workers=0))
    X_train = np.empty((1,13))
    y_train = np.empty((1,))
    for point, target in c0_iter:
        # reshape point to (13,1)
        small_x = point.numpy().reshape(1,13)
        small_y = target.numpy()
        X_train = np.append(X_train, small_x, axis=0)
        y_train = np.append(y_train, small_y, axis=0)
    return X_train[1:], y_train[1:]


def get_client_data(center_id: int, config: dict = None):
    X_train, y_train = _get_np(center_id, train = True)
    X_val, y_val = _get_np(center_id, train = False)
    return X_train, y_train, X_val, y_val

def get_server_data(config: dict = None):
    return _get_np(1, train = False)