import argparse
import csv
import json
import math
import os
import random
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import torch

from ndmd_modules import NeuralDMD
from ndmd_modules import create_test_data
from ndmd_modules import mse
from ndmd_modules import ndmd_training

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sampling-ratio", type = float, default = 0.001)
    parser.add_argument("--zero-tol", type = float, default = 0.001)

    parser.add_argument("--batch-size", type = int, default = 128)
    parser.add_argument("--prediction-steps", type = int, default = 5)
    parser.add_argument("--training-rate", type = float, default = 0.8)
    parser.add_argument("--learning-rate", type = float, default = 1e-3)
    parser.add_argument("--epochs", type = int, default = 10000)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--scheduler-patience", type = int, default = 100)
    parser.add_argument("--early-stopping-patience", type = int, default = 300)
    parser.add_argument("--min-delta", type = float, default = 1e-4)
    parser.add_argument("--min-lr", type = float, default = 1e-6)

    parser.add_argument("--latent-dim", type = int, default = 128)
    parser.add_argument("--koopman-dim", type = int, default = 64)
    parser.add_argument("--lrelu-alpha", type = float, default = 0.01)
    parser.add_argument("--dropout-rate", type = float, default = 0.1)
    parser.add_argument("--regularize-coef", type = float, default = 0.1)
    parser.add_argument("--low-rank", type = float, default = 0.999)

    parser.add_argument("--tune", action = "store_true")
    parser.add_argument("--tune-trials", type = int, default = 20)
    parser.add_argument("--tune-max-epochs", type = int, default = 1000)
    parser.add_argument("--tune-output", type = str, default = "results/cylinder-hparam-search.csv")
    parser.add_argument("--best-config-path", type = str, default = "results/cylinder-best-hparams.json")
    parser.add_argument("--skip-final-train", action = "store_true")

    parser.add_argument("--plot-path", type = str, default = "results/cylinder-prediction-NDMD.pdf")
    parser.add_argument("--plot-state-count", type = int, default = 5)

    return parser.parse_args()

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_sampled_vortacity(
    file_name = "datas/CYLINDER_ALL.mat",
    sampling_ratio = 0.001,
    zero_tol = 1e-12
    ):

    all_data = sp.io.loadmat(file_name)
    data = all_data["VORTALL"]

    if zero_tol < 0:
        raise ValueError("zero_tol must be non-negative")

    row_max_abs = np.max(np.abs(data), axis = 1)
    exact_zero_count = int(np.count_nonzero(row_max_abs == 0.0))
    nonzero_mask = row_max_abs > zero_tol

    nonzero_count = int(np.count_nonzero(nonzero_mask))
    removed_count = data.shape[0] - nonzero_count
    if nonzero_count == 0:
        raise ValueError("all rows are zero after applying zero_tol")

    data = data[nonzero_mask, :]
    print(
        f"removed near-zero rows: {removed_count} / "
        f"remaining rows: {data.shape[0]} / "
        f"exact-zero rows: {exact_zero_count} / "
        f"zero_tol: {zero_tol:.1e}"
        )

    idx = np.arange(0, data.shape[0])
    sample_count = max(1, min(idx.shape[0], int(sampling_ratio*idx.shape[0])))
    sampled = np.random.choice(idx, size = sample_count, replace = False)
    data_sampled = data[sampled, :]
    print(f"sampled rows: {sample_count} / {data.shape[0]}")

    return data_sampled

def normalized_early_stopping_patience(args):

    if args.early_stopping_patience <= 0:
        return None

    return args.early_stopping_patience

def config_from_args(args):

    return {
        "latent_dim": args.latent_dim,
        "koopman_dim": args.koopman_dim,
        "lrelu_alpha": args.lrelu_alpha,
        "dropout_rate": args.dropout_rate,
        "regularize_coef": args.regularize_coef,
        "low_rank": args.low_rank,
        "learning_rate": args.learning_rate
        }

def build_ndmd_model(original_dim, config, prediction_steps):

    return NeuralDMD(
        original_dim = original_dim,
        latent_dim = int(config["latent_dim"]),
        koopman_dim = int(config["koopman_dim"]),
        lrelu_alpha = float(config["lrelu_alpha"]),
        dropout_rate = float(config["dropout_rate"]),
        regularize_coef = float(config["regularize_coef"]),
        low_rank = float(config["low_rank"]),
        prediction_steps = prediction_steps
        )

def train_ndmd_model(
    config,
    args,
    data_train_valid,
    original_dim,
    epochs,
    verbose = True
    ):

    ndmd_model = build_ndmd_model(
        original_dim = original_dim,
        config = config,
        prediction_steps = args.prediction_steps
        )

    return ndmd_training(
        model = ndmd_model,
        data = data_train_valid,
        training_rate = args.training_rate,
        batch_size = args.batch_size,
        learning_rate = float(config["learning_rate"]),
        epochs = epochs,
        prediction_steps = args.prediction_steps,
        scheduler_patience = args.scheduler_patience,
        early_stopping_patience = normalized_early_stopping_patience(args),
        min_delta = args.min_delta,
        min_lr = args.min_lr,
        verbose = verbose,
        return_metrics = True
        )

def sample_log_float(rng, low, high):

    return math.exp(rng.uniform(math.log(low), math.log(high)))

def sample_trial_config(rng):

    latent_dim_choices = [32, 64, 128, 256, 512]
    koopman_dim_choices = [16, 32, 64, 128, 256]

    latent_dim = rng.choice(latent_dim_choices)
    koopman_dim = rng.choice([
        dim for dim in koopman_dim_choices
        if dim <= latent_dim
        ])

    regularize_coef = (
        0.0
        if rng.random() < 0.15
        else sample_log_float(rng, low = 1e-6, high = 1.0)
        )

    return {
        "latent_dim": latent_dim,
        "koopman_dim": koopman_dim,
        "lrelu_alpha": sample_log_float(rng, low = 1e-4, high = 2e-1),
        "dropout_rate": rng.uniform(0.0, 0.5),
        "regularize_coef": regularize_coef,
        "low_rank": rng.uniform(0.90, 0.9999),
        "learning_rate": sample_log_float(rng, low = 1e-5, high = 3e-3)
        }

def format_config(config):

    return (
        f"latent_dim={config['latent_dim']}, "
        f"koopman_dim={config['koopman_dim']}, "
        f"lrelu_alpha={config['lrelu_alpha']:.3g}, "
        f"dropout_rate={config['dropout_rate']:.3g}, "
        f"regularize_coef={config['regularize_coef']:.3g}, "
        f"low_rank={config['low_rank']:.5f}, "
        f"learning_rate={config['learning_rate']:.3g}"
        )

def save_json(path, payload):

    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, indent = 2)

def run_hyperparameter_search(args, data_train_valid, original_dim):

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.tune_output) or ".", exist_ok = True)

    fieldnames = [
        "trial",
        "status",
        "valid_loss",
        "best_epoch",
        "stopped_epoch",
        "final_lr",
        "latent_dim",
        "koopman_dim",
        "lrelu_alpha",
        "dropout_rate",
        "regularize_coef",
        "low_rank",
        "learning_rate",
        "error"
        ]

    best_config = None
    best_record = None

    with open(args.tune_output, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for trial in range(1, args.tune_trials + 1):
            config = sample_trial_config(rng)
            set_random_seed(args.seed + trial)

            print(f"[trial {trial}/{args.tune_trials}] {format_config(config)}")

            try:
                _, metrics = train_ndmd_model(
                    config = config,
                    args = args,
                    data_train_valid = data_train_valid,
                    original_dim = original_dim,
                    epochs = args.tune_max_epochs,
                    verbose = False
                    )
                status = "ok"
                error = ""
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                metrics = {
                    "best_loss": float("inf"),
                    "best_epoch": 0,
                    "stopped_epoch": 0,
                    "final_lr": float(config["learning_rate"])
                    }
                status = "failed"
                error = repr(exc)

            record = {
                "trial": trial,
                "status": status,
                "valid_loss": metrics["best_loss"],
                "best_epoch": metrics["best_epoch"],
                "stopped_epoch": metrics["stopped_epoch"],
                "final_lr": metrics["final_lr"],
                "error": error,
                **config
                }
            writer.writerow(record)
            f.flush()

            if status == "ok":
                print(
                    f"  valid_loss={metrics['best_loss']:.6f} / "
                    f"best_epoch={metrics['best_epoch']} / "
                    f"stopped_epoch={metrics['stopped_epoch']}"
                    )

                if (
                    math.isfinite(metrics["best_loss"])
                    and (
                        best_record is None
                        or metrics["best_loss"] < best_record["valid_loss"]
                        )
                    ):
                    best_config = config
                    best_record = record
            else:
                print(f"  failed: {error}")

    if best_config is None:
        raise RuntimeError("all hyperparameter trials failed")

    payload = {
        "best_trial": best_record["trial"],
        "best_valid_loss": best_record["valid_loss"],
        "best_epoch": best_record["best_epoch"],
        "stopped_epoch": best_record["stopped_epoch"],
        "hyperparameters": best_config,
        "search": {
            "trials": args.tune_trials,
            "max_epochs_per_trial": args.tune_max_epochs,
            "early_stopping_patience": normalized_early_stopping_patience(args),
            "scheduler_patience": args.scheduler_patience,
            "min_delta": args.min_delta,
            "seed": args.seed
            }
        }
    save_json(args.best_config_path, payload)

    print(f"best trial: {best_record['trial']} / valid_loss={best_record['valid_loss']:.6f}")
    print(f"saved search results: {args.tune_output}")
    print(f"saved best config: {args.best_config_path}")

    return best_config, best_record

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

    return float(test_loss.item())

def main():

    args = parse_args()
    set_random_seed(args.seed)

    data_sampled = load_sampled_vortacity(
        sampling_ratio = args.sampling_ratio,
        zero_tol = args.zero_tol
        )
    data_train_valid = data_sampled[:, :int(args.training_rate*data_sampled.shape[1])]
    data_test = data_sampled[:, int(args.training_rate*data_sampled.shape[1]):]

    if args.tune:
        best_config, _ = run_hyperparameter_search(
            args = args,
            data_train_valid = data_train_valid,
            original_dim = data_sampled.shape[0]
            )

        if args.skip_final_train:
            return

        final_config = best_config
        print("training final model with best hyperparameters")
    else:
        final_config = config_from_args(args)

    set_random_seed(args.seed)
    ndmd_model, metrics = train_ndmd_model(
        config = final_config,
        args = args,
        data_train_valid = data_train_valid,
        original_dim = data_sampled.shape[0],
        epochs = args.epochs,
        verbose = True
        )

    print(
        f"selected valid loss: {metrics['best_loss']:.6f} / "
        f"best epoch: {metrics['best_epoch']} / "
        f"stopped epoch: {metrics['stopped_epoch']}"
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
