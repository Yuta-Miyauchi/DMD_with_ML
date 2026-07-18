import argparse
import csv
import os

import matplotlib.pyplot as plt

from ndmd_modules import config_from_args
from ndmd_modules import load_sampled_vortacity
from ndmd_modules import ndmd_test
from ndmd_modules import run_hyperparameter_search
from ndmd_modules import set_random_seed
from ndmd_modules import split_train_valid_test
from ndmd_modules import train_ndmd_model

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sampling-ratio", type = float, default = 0.001)
    parser.add_argument("--zero-tol", type = float, default = 1e-3)

    parser.add_argument("--prediction-steps-start", type = int, default = 1)
    parser.add_argument("--prediction-steps-end", type = int, default = 10)
    parser.add_argument("--batch-size", type = int, default = 128)
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

    parser.add_argument("--no-tune", action = "store_true")
    parser.add_argument("--tune-trials", type = int, default = 20)
    parser.add_argument("--tune-max-epochs", type = int, default = 1000)
    parser.add_argument("--tune-output", type = str, default = "results/cylinder-hparam-search.csv")
    parser.add_argument("--best-config-path", type = str, default = "results/cylinder-best-hparams.json")

    parser.add_argument("--step-results-dir", type = str, default = "results/prediction_steps")
    parser.add_argument("--summary-output", type = str, default = "results/cylinder-prediction-steps.csv")
    parser.add_argument("--test-loss-plot-path", type = str, default = "results/cylinder-test-loss-by-prediction-steps.pdf")
    parser.add_argument("--skip-prediction-plots", action = "store_true")
    parser.add_argument("--plot-state-count", type = int, default = 5)

    return parser.parse_args()

def validate_args(args):

    if args.prediction_steps_start < 1:
        raise ValueError("prediction_steps_start must be at least 1")
    if args.prediction_steps_end < args.prediction_steps_start:
        raise ValueError("prediction_steps_end must be >= prediction_steps_start")

def write_summary_header(path):

    os.makedirs(os.path.dirname(path) or ".", exist_ok = True)
    fieldnames = [
        "prediction_steps",
        "test_loss",
        "search_valid_loss",
        "final_valid_loss",
        "prediction_plot_path",
        "best_epoch",
        "stopped_epoch",
        "latent_dim",
        "koopman_dim",
        "lrelu_alpha",
        "dropout_rate",
        "regularize_coef",
        "low_rank",
        "learning_rate"
        ]

    f = open(path, "w", newline = "", encoding = "utf-8")
    writer = csv.DictWriter(f, fieldnames = fieldnames)
    writer.writeheader()

    return f, writer

def plot_test_loss(results, plot_path):

    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok = True)

    prediction_steps = [row["prediction_steps"] for row in results]
    test_losses = [row["test_loss"] for row in results]

    fig, ax = plt.subplots(figsize = (7, 4.5))
    ax.plot(prediction_steps, test_losses, marker = "o", linewidth = 1.8)
    ax.set_xlabel("prediction_steps")
    ax.set_ylabel("test_loss")
    ax.set_xticks(prediction_steps)
    ax.grid(True, linewidth = 0.3, alpha = 0.5)
    fig.tight_layout()
    plt.savefig(plot_path, format = os.path.splitext(plot_path)[1].lstrip(".") or "pdf")
    plt.close()

    print(f"saved test loss plot: {plot_path}")

def run_one_prediction_step(args, data_train_valid, data_test, original_dim, prediction_steps):

    step_seed_offset = prediction_steps*1000
    set_random_seed(args.seed + step_seed_offset)
    args.prediction_steps = prediction_steps

    if args.no_tune:
        config = config_from_args(args)
        best_record = {
            "valid_loss": None,
            "best_epoch": None,
            "stopped_epoch": None
            }
    else:
        tune_output = os.path.join(
            args.step_results_dir,
            f"prediction_steps_{prediction_steps:02d}_hparam_search.csv"
            )
        best_config_path = os.path.join(
            args.step_results_dir,
            f"prediction_steps_{prediction_steps:02d}_best_hparams.json"
            )
        config, best_record = run_hyperparameter_search(
            args = args,
            data_train_valid = data_train_valid,
            original_dim = original_dim,
            prediction_steps = prediction_steps,
            tune_output = tune_output,
            best_config_path = best_config_path,
            seed_offset = step_seed_offset
            )

    print(f"training final model for prediction_steps={prediction_steps}")
    set_random_seed(args.seed + step_seed_offset)
    ndmd_model, metrics = train_ndmd_model(
        config = config,
        args = args,
        data_train_valid = data_train_valid,
        original_dim = original_dim,
        epochs = args.epochs,
        prediction_steps = prediction_steps,
        verbose = True
        )

    plot_path = None
    if not args.skip_prediction_plots:
        plot_path = os.path.join(
            args.step_results_dir,
            f"prediction_steps_{prediction_steps:02d}_prediction.pdf"
            )

    test_loss = ndmd_test(
        ndmd_model,
        data_test,
        prediction_steps = prediction_steps,
        plot_path = plot_path,
        plot_state_count = args.plot_state_count
        )

    return {
        "prediction_steps": prediction_steps,
        "test_loss": test_loss,
        "search_valid_loss": None if args.no_tune else best_record["valid_loss"],
        "final_valid_loss": metrics["best_loss"],
        "prediction_plot_path": plot_path,
        "best_epoch": metrics["best_epoch"],
        "stopped_epoch": metrics["stopped_epoch"],
        **config
        }

def main():

    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)

    data_sampled = load_sampled_vortacity(
        sampling_ratio = args.sampling_ratio,
        zero_tol = args.zero_tol
        )
    data_train_valid, data_test = split_train_valid_test(
        data_sampled,
        training_rate = args.training_rate
        )
    os.makedirs(args.step_results_dir, exist_ok = True)

    results = []
    summary_file, summary_writer = write_summary_header(args.summary_output)
    try:
        for prediction_steps in range(
            args.prediction_steps_start,
            args.prediction_steps_end + 1
            ):
            print(f"=== prediction_steps={prediction_steps} ===")
            row = run_one_prediction_step(
                args = args,
                data_train_valid = data_train_valid,
                data_test = data_test,
                original_dim = data_sampled.shape[0],
                prediction_steps = prediction_steps
                )
            results.append(row)
            summary_writer.writerow(row)
            summary_file.flush()
    finally:
        summary_file.close()

    plot_test_loss(results, args.test_loss_plot_path)
    print(f"saved summary: {args.summary_output}")

if __name__ == "__main__":
    main()
