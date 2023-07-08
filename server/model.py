import sklearn
import torch

def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)

def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
    """Loads pretrained efficientnet model from torch hub. Replaces final
    classifying layer if classes is specified.

    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model

    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )

    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet
