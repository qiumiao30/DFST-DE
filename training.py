import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import PolynomialLRDecay
from tools import adjust_learning_rate, EarlyStopping
from torch.optim import lr_scheduler
import torch.optim.swa_utils


class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.forecast_criterion2 = nn.KLDivLoss(reduction="batchmean")
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        #self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=5, end_learning_rate=0.0001, power=2)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, verbose=True)
        self.swa_start = 15
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=0.001)

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "val_total": [],
            "val_forecast": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        self.early_stopping = EarlyStopping(patience=3, verbose=True)

        if self.log_tensorboard: # 对应logs文件内容
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """

        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                # inp = x[:,1:]-x[:,:-1]
                # inp = torch.cat([inp, x[:,-1]-x[:,-2]], dim=1)

                preds = self.model(x)

                state = self.model.state_dict()

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                loss = self.forecast_criterion(y, preds)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(loss.item())

            # self.scheduler.step()
            if epoch > self.swa_start:
                self.swa_scheduler.step()
            else:
                self.scheduler.step()

            forecast_b_losses = np.array(forecast_b_losses)
            forecast_epoch_loss = forecast_b_losses.mean()

            self.losses["train_forecast"].append(forecast_epoch_loss)
            # Evaluate on validation set
            forecast_val_loss = "NA"
            if val_loader is not None:
                forecast_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)

                # if forecast_val_loss <= any(self.losses["val_forecast"]):
                #     self.save(f"model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                    )
                s += f" [lr: {self.optimizer.param_groups[0]['lr']:.5f}]"
                s += f" [{epoch_time:.1f}s]"
                print(s)

            # adjust_learning_rate(self.optimizer, epoch, "type1", self.init_lr)
            self.early_stopping(forecast_val_loss, self.model, path=f"{self.dload}/model.pt")
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        if val_loader is None:
            self.save(f"model.pt")

        train_time = int(time.time() - train_start)

        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return forecasting loss, reconstruction loss, total loss
        """
        self.model.eval()

        forecast_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if preds.ndim == 3:
                    preds = preds.squeeze(1)
                if y.ndim == 3:
                    y = y.squeeze(1)

                forecast_loss = self.forecast_criterion(y, preds)# nn.MSE

                forecast_losses.append(forecast_loss.item())

        forecast_losses = np.array(forecast_losses)
        forecast_loss = (forecast_losses + 1e-8).mean()

        return forecast_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
