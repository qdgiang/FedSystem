import pickle
import os
import sys
sys.path.append(os.path.abspath(""))
from dataset.advanced_benchmark.constants import MEAN, STD
from dataset.advanced_benchmark.datasets import DATASETS
import torch
from typing import List, Type, Dict
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


CURRENT_DATASET = "cifar10"

def get_client_data(partition_id: int, config: dict):
    print(config)
    dir = os.path.abspath("")
    name = config["advanced_name"]
    try:
        partition_path = f"{dir}/dataset/advanced_benchmark/{name}.pkl"
        with open(partition_path, "rb") as f:
            partition = pickle.load(f)
    except:
        raise FileNotFoundError(f"Dataset not found")

    data_indices: List[List[int]] = partition["data_indices"]

    general_data_transform = transforms.Compose(
            [transforms.Normalize(MEAN[CURRENT_DATASET], STD[CURRENT_DATASET])]
        )
    general_target_transform = transforms.Compose([])
    train_data_transform = transforms.Compose([])
    train_target_transform = transforms.Compose([])

    my_dataset = DATASETS[CURRENT_DATASET](
            root=f"{dir}/dataset/advanced_benchmark/{CURRENT_DATASET}",
            args=None,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

    trainset: Subset = Subset(
        dataset=my_dataset, 
        indices=data_indices[partition_id]["train"]
    )
    valset: Subset = Subset(
        dataset=my_dataset, 
        indices=data_indices[partition_id]["test"]
    )
    trainloader = DataLoader(trainset, batch_size = 32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=True)
    return trainloader, None, valloader, None
    

def get_server_data(config: dict):
    dir = os.path.abspath("")
    name = config["advanced_name"]
    try:
        partition_path = f"{dir}/data/dataset/advanced_benchmark/{name}.pkl"
        with open(partition_path, "rb") as f:
            partition = pickle.load(f)
    except:
        raise FileNotFoundError(f"Dataset not found")

    data_indices: List[List[int]] = partition["data_indices"]

    general_data_transform = transforms.Compose(
            [transforms.Normalize(MEAN[CURRENT_DATASET], STD[CURRENT_DATASET])]
        )
    general_target_transform = transforms.Compose([])
    train_data_transform = transforms.Compose([])
    train_target_transform = transforms.Compose([])
    
    my_dataset = DATASETS[CURRENT_DATASET](
        root=f"{dir}/dataset/advanced_benchmark/{CURRENT_DATASET}",
        args=None,
        general_data_transform=general_data_transform,
        general_target_transform=general_target_transform,
        train_data_transform=train_data_transform,
        train_target_transform=train_target_transform,
    )

    testdata_indices = []
    for indices in data_indices:
        testdata_indices.extend(indices["test"])
    
    testset: Subset = Subset(
        dataset=my_dataset,
        indices=testdata_indices
    )
    return DataLoader(testset, batch_size=32, shuffle=True), None