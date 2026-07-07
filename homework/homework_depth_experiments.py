from fully_connected_basics.datasets import get_mnist_loaders, get_cifar_loaders
from homework.experiment_utils import run_experiment, save_plot

configs_dropouts = {
    "5_layers_dropout)": [
            {"type": "linear", "size": 1024},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 512},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 256},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 128},
            {"type": "dropout", "rate": 0.2},
            {"type": "linear", "size": 64}
        ],

    "5_layers_batchnorm)": [
            {"type": "linear", "size": 1024},
            {"type": "batch_norm"},
            {"type": "linear", "size": 512},
            {"type": "batch_norm"},
            {"type": "linear", "size": 256},
            {"type": "batch_norm"},
            {"type": "linear", "size": 128},
            {"type": "batch_norm"},
            {"type": "linear", "size": 64},
            {"type": "batch_norm"}
        ]
    }

configs_basic = {
        "1_layer": [
            {"type": "linear", "size": 128}
        ],
        "2_layers": [
             {"type": "linear", "size": 256},
             {"type": "linear", "size": 128}
        ],
        "3_layers": [
             {"type": "linear", "size": 512},
             {"type": "linear", "size": 256},
             {"type": "linear", "size": 128}
        ],
        "5_layers": [
             {"type": "linear", "size": 1024},
             {"type": "linear", "size": 512},
             {"type": "linear", "size": 256},
             {"type": "linear", "size": 128},
             {"type": "linear", "size": 64}
        ],
        "7_layers": [
             {"type": "linear", "size": 1024},
             {"type": "linear", "size": 512},
             {"type": "linear", "size": 256},
             {"type": "linear", "size": 128},
             {"type": "linear", "size": 64},
             {"type": "linear", "size": 32},
             {"type": "linear", "size": 16}
        ]
    }


if __name__ == "__main__":
    results_mnist = run_experiment(configs_basic, get_loaders = get_mnist_loaders)
    results_cifar = run_experiment(configs_basic, get_loaders = get_mnist_loaders)
    results_dropout = run_experiment(configs_dropouts)

    save_plot(results_mnist + results_cifar + results_dropout)