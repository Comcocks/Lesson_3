import time
import torch
from matplotlib import pyplot as plt
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.datasets import get_mnist_loaders
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import count_parameters
from homework.experiment_utils import save_plot

base = [
    {"type": "linear", "size": 256}, {"type": "relu"},
    {"type": "linear", "size": 128}, {"type": "relu"},
    {"type": "linear", "size": 64}, {"type": "relu"},
]


def add_dropout(rate):
    result = []
    for layer in base:
        result.append(layer)
        if layer.get("type") == "relu":
            result.append({"type": "dropout", "rate": rate})
    return result

def add_batchnorm(default):
    result = []
    for layer in default:
        result.append(layer)
        if layer.get("type") == "linear":
            result.append({"type": "batch_norm"})
    return result


configs = {
    "no_regularization": base,
    "dropout=0.1": add_dropout(0.1),
    "dropout=0.3": add_dropout(0.3),
    "dropout=0.5": add_dropout(0.5),
    "batchnorm": add_batchnorm(base),
    "dropout_batchnorm": add_batchnorm(add_dropout(0.3)),
    "l2_regularization": base
}


def regularization_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    results = {}

    for n, l in configs.items():
        print(f"\nTraining: {n}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=l).to(device)
        params = count_parameters(model)

        start = time.time()
        stats = train_model(model, train_loader, test_loader, epochs=10, device=str(device), weight_decay=1e-4 if "l2" in n else 0.0)
        end = time.time() - start

        weights = []
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                weights.append(m.weight)

        results[n] = {
            "history": stats,
            "params": params,
            "time": end,
            "weights": weights
        }

    return results


if __name__ == "__main__":
    results = regularization_experiments()
    save_plot(results)

    for n, r in results.items():
        weights = r.get("weights", [])

        if not weights:
            continue

        for i, w in enumerate(weights):
            plt.figure()
            plt.hist(w.flatten().detach().cpu().numpy(), bins=50)
            plt.xlabel("Weight")
            plt.ylabel("Frequency")
            plt.title(f"{n} - {i} Layer")
            plt.tight_layout()
            plt.savefig(f"histograms/{n}_layer{i}.png")
            plt.close()