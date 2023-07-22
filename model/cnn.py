import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def init_model(model_config: dict) -> Net:
    return Net()

def get_parameters(model: Net) -> list:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: Net, parameters: list) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def fit(
    model: Net, X_train: DataLoader, y_train: DataLoader, model_config: dict
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    epochs = model_config["epochs"]
    verbose = model_config["verbose"]
    DEVICE = torch.device("cpu") if model_config["device"] == "cpu" else torch.device("cuda:0")
    
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in X_train:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(X_train.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        return get_parameters(model), len(X_train.dataset), {"accuracy": epoch_acc}

def evaluate(
    model: Net, X_test: DataLoader, y_test: DataLoader, model_config: dict
):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    epochs = model_config["epochs"]
    verbose = model_config["verbose"]
    DEVICE = torch.device("cpu") if model_config["cuda"] == False else torch.device("cuda:0")

    with torch.no_grad():
        for images, labels in X_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(X_test.dataset)
    accuracy = correct / total
    return loss, len(X_test.dataset), {"accuracy": accuracy}