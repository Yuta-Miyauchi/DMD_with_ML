import csv
import json
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import torch

class NeuralDMD(torch.nn.Module):
    def __init__(
        self,
        original_dim,
        latent_dim,
        koopman_dim,
        lrelu_alpha = 0.01,
        dropout_rate = 0.1,
        regularize_coef = 0.1,
        low_rank = 0.999,
        prediction_steps = 5
        ):

        super(NeuralDMD, self).__init__()

        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.koopman_dim = koopman_dim
        self.lrelu_alpha = lrelu_alpha
        self.dropout_rate = dropout_rate
        self.regularize_coef = regularize_coef
        self.low_rank = low_rank
        self.prediction_steps = prediction_steps

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.original_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.koopman_dim, bias = False)
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.koopman_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.latent_dim, bias = False),
            torch.nn.LeakyReLU(self.lrelu_alpha),
            torch.nn.Dropout(self.dropout_rate),

            torch.nn.Linear(self.latent_dim, self.original_dim, bias = False)
            )

    def forward(self, x):

        if self.training:

            z = self.encoder(x)

            z_prediction, regularize_term = self._dmd(z)

            x_prediction = self.decoder(z_prediction)

            return x_prediction, regularize_term

        else:

            z = self.encoder(x)

            z = z.T
            z_prediction = z
            for i in range(1, self.prediction_steps):
                z_prediction = torch.cat(
                    [z_prediction, self.U@torch.pow(self.Atilde, i)@self.U.T@z],
                    dim = 1
                    )
            z_prediction = z_prediction.T

            x_prediction = self.decoder(z_prediction)

            return x_prediction

    def _low_rank_approximation(self, s, low_rank):

        ratio_s = s/s.sum()
        cumulative_s = torch.cumsum(ratio_s, dim = 0)
        idx = torch.nonzero(cumulative_s >= low_rank, as_tuple = False)
        if len(idx) == 0:
            return len(s)
        else:
            return idx[0].item() + 1

    def _dmd(self, x):

        x = x.T

        x1 = x[:, :(x.shape[1]//2)]
        x2 = x[:, (x.shape[1]//2):]

        U, s, Vh = torch.linalg.svd(x1, full_matrices = False)
        V = Vh.T

        r = self._low_rank_approximation(s, low_rank = self.low_rank)
        self.U = U[:, :r]
        s = s[:r]
        V = V[:, :r]

        s_inv = 1/s
        S_inv = torch.diag(s_inv)

        self.Atilde = self.U.T@x2@V@S_inv
        regularize_term = self.regularize_coef*torch.linalg.matrix_norm(self.Atilde, ord = 2)

        x_prediction = x1
        for i in range(1, self.prediction_steps):
            x_prediction = torch.cat(
                [x_prediction, self.U@torch.pow(self.Atilde, i)@self.U.T@x1],
                dim = 1
                )

        return x_prediction.T, regularize_term

def create_train_data(
    data,
    batch_size = 128,
    prediction_steps = 5
    ):

    t_idx = np.arange(data.shape[1] - prediction_steps)
    if batch_size >= data.shape[1]:
        t_sampled = np.concatenate([[0], np.random.choice(t_idx, size = t_idx.shape[0], replace = False)])
    else:
        t_sampled = np.concatenate([[0], np.random.choice(t_idx, size = batch_size, replace = False)])

    x = np.transpose(np.concatenate([data[:, t_sampled], data[:, t_sampled + 1]], axis = 1))

    y = data[:, t_sampled]
    for i in range(1, prediction_steps):
        y = np.concatenate([y, data[:, t_sampled + i]], axis = 1)
    y = np.transpose(y)

    return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)

def create_valid_data(data, prediction_steps = 5):

    t_idx = np.arange(data.shape[1] - prediction_steps)

    x = np.transpose(data[:, t_idx])

    y = data[:, t_idx]
    for i in range(1, prediction_steps):
        y = np.concatenate([y, data[:, t_idx + i]], axis = 1)
    y = np.transpose(y)

    return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)

def create_test_data(data, prediction_steps = 5):

    t_idx = np.arange(data.shape[1] - prediction_steps)

    x = np.transpose(data[:, t_idx])

    y = data[:, t_idx]
    for i in range(1, prediction_steps):
        y = np.concatenate([y, data[:, t_idx + i]], axis = 1)
    y = np.transpose(y)

    return torch.tensor(x, dtype = torch.float), torch.tensor(y, dtype = torch.float)

def mse(y_prediction, y_true):

    err = torch.norm(y_prediction - y_true, dim = 1)
    mse = err.mean()

    return mse

def ndmd_training(
    model,
    data,
    training_rate = 0.8,
    batch_size = 128,
    learning_rate = 1e-3,
    epochs = 1000,
    prediction_steps = 5,
    scheduler_patience = 100,
    early_stopping_patience = None,
    min_delta = 1e-4,
    min_lr = 1e-6,
    verbose = True,
    return_metrics = False
    ):

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_dmd_state = None
    stopped_epoch = epochs
    epochs_without_improvement = 0

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        factor = 0.1,
        patience = scheduler_patience,
        threshold = min_delta,
        min_lr = min_lr
        )

    for epoch in range(epochs):

        x, y = create_train_data(
            data[:, :int(training_rate*data.shape[1])],
            batch_size = batch_size,
            prediction_steps = prediction_steps
            )

        model.train()
        y_prediction, regularize_term = model(x)
        train_loss = mse(y_prediction, y) + regularize_term
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        x, y = create_valid_data(
            data[:, int(training_rate*data.shape[1]):],
            prediction_steps = prediction_steps
            )

        model.eval()
        with torch.no_grad():
            y_prediction = model(x)
            valid_loss = mse(y_prediction, y)
        valid_loss_value = float(valid_loss.item())
        scheduler.step(valid_loss_value)
        if valid_loss_value < best_loss - min_delta:
            best_loss = valid_loss_value
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
                }
            best_dmd_state = {
                "U": model.U.detach().clone(),
                "Atilde": model.Atilde.detach().clone()
                }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]

        if verbose and (epoch == 0 or (epoch + 1)%100 == 0):
            print(f"epoch {epoch + 1: 4} / train loss: {train_loss:.6f} / valid loss: {valid_loss:.6f} / lr: {current_lr:.2e}")

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
            ):
            stopped_epoch = epoch + 1
            if verbose:
                print(f"early stopping at epoch {stopped_epoch} / best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a finite validation loss")

    if verbose:
        print(f"best loss: {best_loss}")
    model.load_state_dict(best_state)
    model.U = best_dmd_state["U"]
    model.Atilde = best_dmd_state["Atilde"]

    metrics = {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "final_lr": current_lr
        }

    if return_metrics:
        return model, metrics

    return model

def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_sampled_vortacity(
    file_name = "datas/CYLINDER_ALL.mat",
    sampling_ratio = 0.001,
    zero_tol = 1e-3
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

def split_train_valid_test(data, training_rate = 0.8):

    test_start = int(training_rate*data.shape[1])

    return data[:, :test_start], data[:, test_start:]

def normalized_early_stopping_patience(early_stopping_patience):

    if early_stopping_patience <= 0:
        return None

    return early_stopping_patience

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
    prediction_steps = None,
    verbose = True
    ):

    if prediction_steps is None:
        prediction_steps = args.prediction_steps

    ndmd_model = build_ndmd_model(
        original_dim = original_dim,
        config = config,
        prediction_steps = prediction_steps
        )

    return ndmd_training(
        model = ndmd_model,
        data = data_train_valid,
        training_rate = args.training_rate,
        batch_size = args.batch_size,
        learning_rate = float(config["learning_rate"]),
        epochs = epochs,
        prediction_steps = prediction_steps,
        scheduler_patience = args.scheduler_patience,
        early_stopping_patience = normalized_early_stopping_patience(
            args.early_stopping_patience
            ),
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

def run_hyperparameter_search(
    args,
    data_train_valid,
    original_dim,
    prediction_steps = None,
    tune_output = None,
    best_config_path = None,
    seed_offset = 0
    ):

    if prediction_steps is None:
        prediction_steps = args.prediction_steps
    if tune_output is None:
        tune_output = args.tune_output
    if best_config_path is None:
        best_config_path = args.best_config_path

    rng = random.Random(args.seed + seed_offset)
    os.makedirs(os.path.dirname(tune_output) or ".", exist_ok = True)

    fieldnames = [
        "trial",
        "status",
        "prediction_steps",
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

    with open(tune_output, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()

        for trial in range(1, args.tune_trials + 1):
            config = sample_trial_config(rng)
            set_random_seed(args.seed + seed_offset + trial)

            print(
                f"[prediction_steps={prediction_steps} / "
                f"trial {trial}/{args.tune_trials}] {format_config(config)}"
                )

            try:
                _, metrics = train_ndmd_model(
                    config = config,
                    args = args,
                    data_train_valid = data_train_valid,
                    original_dim = original_dim,
                    epochs = args.tune_max_epochs,
                    prediction_steps = prediction_steps,
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
                "prediction_steps": prediction_steps,
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
        "prediction_steps": prediction_steps,
        "best_trial": best_record["trial"],
        "best_valid_loss": best_record["valid_loss"],
        "best_epoch": best_record["best_epoch"],
        "stopped_epoch": best_record["stopped_epoch"],
        "hyperparameters": best_config,
        "search": {
            "trials": args.tune_trials,
            "max_epochs_per_trial": args.tune_max_epochs,
            "early_stopping_patience": normalized_early_stopping_patience(
                args.early_stopping_patience
                ),
            "scheduler_patience": args.scheduler_patience,
            "min_delta": args.min_delta,
            "seed": args.seed + seed_offset
            }
        }
    save_json(best_config_path, payload)

    print(f"best trial: {best_record['trial']} / valid_loss={best_record['valid_loss']:.6f}")
    print(f"saved search results: {tune_output}")
    print(f"saved best config: {best_config_path}")

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

    print(f"test loss: {test_loss:.6f}")

    if plot_path is not None:
        plot_prediction_segments(
            data = data,
            y_prediction = y_prediction,
            prediction_steps = prediction_steps,
            plot_path = plot_path,
            plot_state_count = plot_state_count
            )

    return float(test_loss.item())
