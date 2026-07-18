import argparse

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

    parser.add_argument("--tune-trials", type = int, default = 20)
    parser.add_argument("--tune-max-epochs", type = int, default = 1000)
    parser.add_argument("--tune-output", type = str, default = "results/cylinder-hparam-search.csv")
    parser.add_argument("--best-config-path", type = str, default = "results/cylinder-best-hparams.json")
    parser.add_argument("--skip-final-train", action = "store_true")

    parser.add_argument("--plot-path", type = str, default = "results/cylinder-prediction-NDMD-tuned.pdf")
    parser.add_argument("--plot-state-count", type = int, default = 5)

    return parser.parse_args()

def main():

    args = parse_args()
    set_random_seed(args.seed)

    data_sampled = load_sampled_vortacity(
        sampling_ratio = args.sampling_ratio,
        zero_tol = args.zero_tol
        )
    data_train_valid, data_test = split_train_valid_test(
        data_sampled,
        training_rate = args.training_rate
        )

    best_config, _ = run_hyperparameter_search(
        args = args,
        data_train_valid = data_train_valid,
        original_dim = data_sampled.shape[0]
        )

    if args.skip_final_train:
        return

    print("training final model with best hyperparameters")
    set_random_seed(args.seed)
    ndmd_model, metrics = train_ndmd_model(
        config = best_config,
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
