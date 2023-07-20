from flamby.datasets.fed_heart_disease import FedHeartDisease
from torch.utils.data import DataLoader as dl
import numpy as np



def _get_np(center_id: int, train: bool, config: dict = None):
    center = FedHeartDisease(center=center_id, train=train)
    if config["split"] == True:
        c0_iter = iter(dl(center, batch_size=None, num_workers=0))
        X_train = np.empty((1,13))
        y_train = np.empty((1,))
        flag = True
        for point, target in c0_iter:
            # reshape point to (13,1)
            if flag == True:
                print(point.shape)
                print(target.shape)
                print(point)
                print(target)
                flag = False
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


def get_client_data(center_id: int, config: dict = None):
    X_train, y_train = _get_np(center_id, train = True, config = config)
    print(len(X_train))
    X_val, y_val = _get_np(center_id, train = False, config = config)
    print(len(X_val))
    return X_train, y_train, X_val, y_val

def get_server_data(config: dict = None):
    return _get_np(3, train = False)

if __name__ == "__main__":
    config = {"split": True}
    X_train, y_train, X_val, y_val = get_client_data(0, config)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_train[0])
    print(y_train[0])
    print(X_val[0])
    print(y_val[0])
    print(X_train[0].shape)
    print(y_train[0].shape)
    print(X_val[0].shape)
    print(y_val[0].shape)
    X_train, y_train, X_val, y_val = get_server_data(config)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_train[0])
    print(y_train[0])
    print(X_val[0])
    print(y_val[0])