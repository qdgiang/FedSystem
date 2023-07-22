from torch.nn import Module, Linear, ReLU, Sigmoid
import torch
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from collections import OrderedDict

class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
 
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        return X
    

def init_model(model_config: dict = None) -> MLP:
    return MLP(model_config["n_features"])

def get_parameters(model: MLP) -> list:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: MLP, parameters: list) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def fit(model: MLP, X_train, y_train, model_config: dict = None):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
            outputs = torch.round(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
        if verbose == True:
            print(f"Epoch {epoch}: loss = {epoch_loss / total}, accuracy = {correct / total}")
    return get_parameters(model), len(X_train), {"accuracy": correct / total}

def evaluate(model: MLP, X_test: torch.Tensor, y_test: torch.Tensor, model_config: dict = None):
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    epochs = model_config["epochs"]
    verbose = model_config["verbose"]
    DEVICE = torch.device("cpu") if model_config["device"] == "cpu" else torch.device("cuda:0")

    with torch.no_grad():
        for images, labels in X_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            outputs = torch.round(outputs)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()
        if verbose:
            print(f"Test loss = {loss / total}, accuracy = {correct / total}")
    loss = loss / total
    loss = loss.item()
    return loss, len(X_test), {"accuracy": correct / total}

