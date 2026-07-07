import time
import torch
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import count_parameters


configs = {
        "tight": [
            {"type": "linear", "size": 64}, {"type": "relu"},
            {"type": "linear", "size": 32}, {"type": "relu"},
            {"type": "linear", "size": 16}, {"type": "relu"},
        ],
        "normal": [
            {"type": "linear", "size": 256}, {"type": "relu"},
            {"type": "linear", "size": 128}, {"type": "relu"},
            {"type": "linear", "size": 64}, {"type": "relu"},
        ],
        "big": [
            {"type": "linear", "size": 1024}, {"type": "relu"},
            {"type": "linear", "size": 512}, {"type": "relu"},
            {"type": "linear", "size": 256}, {"type": "relu"},
        ],
        "large": [
            {"type": "linear", "size": 2048}, {"type": "relu"},
            {"type": "linear", "size": 1024}, {"type": "relu"},
            {"type": "linear", "size": 512}, {"type": "relu"},
        ],
    }


def regularization_experiments():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    results = {}

    for name, layers in configs.items():
        print(f"\nTraining: {name}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=layers).to(device)
        params = count_parameters(model)

        weight_decay = 1e-4 if "l2" in name else 0.0

        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=10, device=str(device), weight_decay=weight_decay)
        duration = time.time() - start

        weights = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                weights.append(m.weight)

        results[name] = {
            "history": history,
            "params": params,
            "time": duration,
            "weights": weights
        }

    return results


if __name__ == "__main__":
    results = regularization_experiments()