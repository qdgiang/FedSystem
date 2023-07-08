import flwr as fl
import model.LogisticRegression as model_chosen
import argparse
import torch
import data

class MyClient(fl.client.NumPyClient):
    def __init__(
        self,
        X_train,
        y_train,
        X_eval,
        y_eval,
        device: str,
        model,
        model_name: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.validation_split = validation_split
        self.model = model
        self.model_name = model_name

    def fit(self, parameters, config):
        model_chosen.set_model_params(self.model, parameters)
        print(config)
        # if self.model_name == 'LogisticRegression':
        # import fit function from model/LogisticRegression.py
        #if self.model_name == 'LogisticRegression':
        model_chosen.train(self.model, self.X_train, self.y_train, epochs=1)
    def evaluate(self, parameters, config):
        model_chosen.set_model_params(self.model, parameters)
        print(config)
        model_chosen.evaluate(self.model, self.X_eval, self.y_eval)

    def get_parameters(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--use_cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--partition_id",
        type=int,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    # Load a subset of CIFAR-10 to simulate the local data partition
    (X_train, y_train), (X_eval, y_eval) = data.MNIST.MNIST_DATA(args.partition_id).get_data()

    # Start Flower client
    client = MyClient(X_train= X_train, 
                      y_train= y_train, 
                      X_eval= X_eval, 
                      y_eval= y_eval, 
                      device= device, 
                      model=None,
                      model_chosen, 
                      "LogisticRegression")

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
if __name__ == "__main__":
    main()