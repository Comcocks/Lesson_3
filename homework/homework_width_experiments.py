import numpy
from matplotlib import pyplot as plt
from homework.experiment_utils import run_experiment, save_plot

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

def plot_heatmap(results):
    models = list(results.keys())
    accs = [results[m]['history']['test_accs'][-1] for m in models]

    fig, ax = plt.subplots(figsize=(10, 2))
    data = numpy.array(accs).reshape(1, -1)

    ax.set_xticks(numpy.arange(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks([0])
    ax.set_yticklabels(["Test Accuracy"])

    for i in range(1):
        for j in range(len(models)):
            text = ax.text(j, i, f"{data[i, j]:.4f}",
                           ha="center", va="center", color="black")

    plt.title("Heatmap")
    plt.tight_layout()

    plt.savefig(f"plots/width_heatmap.png")
    plt.close()


if __name__ == "__main__":
    results = run_experiment(configs)
    save_plot(results)
    plot_heatmap(results)
