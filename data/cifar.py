import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import torch

BATCH_SIZE = 32


def load_cifar(NUM_CLIENTS = 5):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("../data/dataset/cifar", train=True, download=True, transform=transform)

    # Split training set into partitions
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    return datasets

def get_client_data(partition_id: int, config: dict = None):
    datasets = load_cifar()
    trainset = datasets[partition_id]
    len_val = len(trainset) // 10  # 10% validation set
    len_train = len(trainset) - len_val
    lengths = [len_train, len_val]
    ds_train, ds_val = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    trainloader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(ds_val, batch_size=BATCH_SIZE)
    return trainloader, None, valloader, None

def get_server_data(config: dict = None):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return testloader


