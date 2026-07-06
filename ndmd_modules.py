import numpy as np
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

    """
    data.shape = (data dimension, time length)
    => x.shape = (sampled time length*2, data dimension)
       y.shape = (sampled time length*prediction steps, data dimension)
    """

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
    prediction_steps = 5
    ):

    best_loss = float("inf")

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = "min",
        factor = 0.1,
        patience = 100,
        threshold = 1e-4,
        min_lr = 1e-6
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
        scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = model.state_dict()

        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == 0 or (epoch + 1)%100 == 0:
            print(f"epoch {epoch + 1: 4} / train loss: {train_loss:.6f} / valid loss: {valid_loss:.6f} / lr: {current_lr:.2e}")

    print(f"best loss: {best_loss}")
    model.load_state_dict(best_state)

    return model
