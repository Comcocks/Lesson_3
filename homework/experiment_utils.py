import time

import torch
from matplotlib import pyplot as plt

from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.trainer import train_model
from fully_connected_basics.utils import count_parameters


def run_experiment(config, loader = "mnist"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64) if loader == "mnist" else get_cifar_loaders(batch_size=64)
    results = {}

    for n, l in config.items():
        print(f"\nTraining: {n}")
        model = FullyConnectedModel(input_size=784, num_classes=10, layers=l).to(device)
        params = count_parameters(model)

        start = time.time()
        stats = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        end = time.time() - start

        results[f"{n}_{loader}"] = {
            "history": stats,
            "params": params,
            "time": end
        }
    return results


def save_plot(results):
    for n, r in results.items():
        print(f"Params: {r['params']}\nTime: {r['time']:.4f}\n")
        history = r['history']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['test_losses'], label='Test Loss')
        ax1.set_title(f'{n} train time: {r['time']:.4f}')
        ax1.legend()

        ax2.plot(history['train_accs'], label='Train Acc')
        ax2.plot(history['test_accs'], label='Test Acc')
        ax2.set_title(f'{n} - Accuracy')
        ax2.legend()

        plt.suptitle(n)
        plt.tight_layout()

        plt.savefig(f"plots/{n}.png")
        plt.close()