import argparse
import os
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import torch

from ndmd_modules import NeuralDMD
from ndmd_modules import create_train_data
from ndmd_modules import create_valid_data
from ndmd_modules import create_test_data
from ndmd_modules import mse
from ndmd_modules import ndmd_training

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sampling-ratio", type = float, default = 0.0001)

    parser.add_argument("--batch-size", type = int, default = 128)
    parser.add_argument("--prediction-steps", type = int, default = 5)
    parser.add_argument("--training-rate", type = float, default = 0.8)
    parser.add_argument("--learning-rate", type = float, default = 1e-3)
    parser.add_argument("--epochs", type = int, default = 1000)

    parser.add_argument("--latent-dim", type = int, default = 128)
    parser.add_argument("--koopman-dim", type = int, default = 64)
    parser.add_argument("--lrelu-alpha", type = float, default = 0.01)
    parser.add_argument("--dropout-rate", type = float, default = 0.1)
    parser.add_argument("--regularize-coef", type = float, default = 0.1)
    parser.add_argument("--low-rank", type = float, default = 0.999)
    parser.add_argument("--plot-path", type = str, default = "results/cylinder-prediction-NDMD.pdf")
    parser.add_argument("--plot-state-count", type = int, default = 5)

    return parser.parse_args()

def load_sampled_vortacity(file_name = "datas/CYLINDER_ALL.mat", sampling_ratio = 0.001):

    all_data = sp.io.loadmat(file_name)
    data = all_data["VORTALL"]

    idx = np.arange(0, data.shape[0])
    sampled = np.random.choice(idx, size = int(sampling_ratio*idx.shape[0]), replace = False)
    data_sampled = data[sampled, :]

    return data_sampled

def plot_prediction_segments(
    data,
    y_prediction,
    prediction_steps = 5,
    plot_path = "results/cylinder-prediction-NDMD.pdf",
    plot_state_count = 5
    ):

    num_start_points = data.shape[1] - prediction_steps
    plot_state_count = max(1, min(plot_state_count, data.shape[0]))
    prediction_segments = (
        y_prediction
        .detach()
        .cpu()
        .numpy()
        .reshape(prediction_steps, num_start_points, data.shape[0])
        .transpose(1, 0, 2)
        )

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok = True)

    fig, axes = plt.subplots(
        plot_state_count,
        1,
        figsize = (10, 2.8*plot_state_count),
        sharex = True
        )
    if plot_state_count == 1:
        axes = [axes]

    for state_idx, ax in enumerate(axes):
        ax.plot(
            np.arange(data.shape[1]),
            data[state_idx, :],
            color = "black",
            linewidth = 1.2,
            label = "true"
            )

        for start_idx in range(num_start_points):
            time_idx = np.arange(start_idx, start_idx + prediction_steps)
            ax.plot(
                time_idx,
                prediction_segments[start_idx, :, state_idx],
                color = "tab:orange",
                linewidth = 0.8,
                alpha = 0.45,
                label = "NDMD prediction" if start_idx == 0 else None
                )

        ax.set_ylabel(f"state {state_idx + 1}")
        ax.grid(True, linewidth = 0.3, alpha = 0.4)
        ax.legend(loc = "best")

    axes[-1].set_xlabel("test time index")
    fig.suptitle(f"Cylinder Flow NDMD Prediction ({prediction_steps}-step segments)")
    plt.tight_layout()
    plt.savefig(plot_path, format = os.path.splitext(plot_path)[1].lstrip(".") or "pdf")
    plt.close()

    print(f"saved plot: {plot_path}")

def ndmd_test(
    model,
    data,
    prediction_steps = 5,
    plot_path = "results/cylinder-prediction-NDMD.pdf",
    plot_state_count = 5
    ):

    x, y = create_test_data(data, prediction_steps = prediction_steps)

    model.eval()
    with torch.no_grad():
        y_prediction = model(x)
    test_loss = mse(y_prediction, y)

    print(f'test loss: {test_loss:.6f}')
    plot_prediction_segments(
        data = data,
        y_prediction = y_prediction,
        prediction_steps = prediction_steps,
        plot_path = plot_path,
        plot_state_count = plot_state_count
        )

def main():

    args = parse_args()
    
    data_sampled = load_sampled_vortacity(sampling_ratio = args.sampling_ratio)
    data_train_valid = data_sampled[:, :int(args.training_rate*data_sampled.shape[1])]
    data_test = data_sampled[:, int(args.training_rate*data_sampled.shape[1]):]

    ndmd_model = NeuralDMD(
        original_dim = data_sampled.shape[0],
        latent_dim = args.latent_dim,
        koopman_dim = args.koopman_dim,
        lrelu_alpha = args.lrelu_alpha,
        dropout_rate = args.dropout_rate,
        regularize_coef = args.regularize_coef,
        low_rank = args.low_rank, 
        prediction_steps = args.prediction_steps
        )

    ndmd_model = ndmd_training(
        model = ndmd_model,
        data = data_train_valid,
        training_rate = args.training_rate,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        epochs = args.epochs,
        prediction_steps = args.prediction_steps
    )

    ndmd_test(
        ndmd_model,
        data_test,
        prediction_steps = args.prediction_steps,
        plot_path = args.plot_path,
        plot_state_count = args.plot_state_count
        )

if __name__ == "__main__":
    main()
