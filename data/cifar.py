import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torch
import os
import pickle
BATCH_SIZE = 32


def load_cifar(NUM_CLIENTS = 5):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(f"{os.path.dirname(__file__)}/dataset/cifar", train=True, download=True, transform=transform)

    # Split training set into partitions
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    pickle_path = f"{os.path.dirname(__file__)}/dataset/cifar/partitions.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(datasets, f)
    return datasets

def get_client_data(partition_id: int, config: dict):
    #datasets = load_cifar()
    pickle_path = f"{os.path.dirname(__file__)}/dataset/cifar/partitions.pkl"
    with open(pickle_path, 'rb') as f:
        datasets = pickle.load(f)
    trainset = datasets[partition_id]
    len_val = len(trainset) // 10  # 10% validation set
    len_train = len(trainset) - len_val
    lengths = [len_train, len_val]
    ds_train, ds_val = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    
    # if partition_id == 3, randomized the labels of the training set
    
    #if partition_id in [3,4]:
    #    print("Randomized the labels of the training set")
    #    ds_train.dataset.dataset.targets = torch.randint(0, 10, (50000,))
    
    # if partition_id == 4, set all labels of the training set to 5
    #if partition_id in [3,4]:
    #    print("Set all labels of the training set to 5")
    #    ds_train.dataset.dataset.targets = torch.ones(50000, dtype=torch.int64) * 5
        #print(ds_train.dataset.dataset.targets)


    trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(ds_val, batch_size=BATCH_SIZE)
    return trainloader, None, valloader, None

def get_server_data(config: dict):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return testloader, None


if __name__ == "__main__":
    load_cifar(5)