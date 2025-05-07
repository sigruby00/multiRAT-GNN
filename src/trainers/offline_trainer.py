import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import HuberLoss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
import time
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from .base_trainer import ModelTrainer
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.metrics import r2_score

class OfflineTrainer(ModelTrainer):
    def __init__(self, model, lr=0.01, writer=None, patience=100):
        super().__init__(model, lr, writer)
        self.model = model
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.writer = writer
        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train(self, dataset, ext_train_loader = None, ext_val_loader = None, num_epochs=100, batch_size=16, verbose=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if ext_train_loader is None:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            train_loader = ext_train_loader

        if ext_val_loader is None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        else:
            val_loader = ext_val_loader

        train_rmse_list, train_mae_list, train_r2_list = [], [], []
        val_rmse_list, val_mae_list, val_r2_list = [], [], []

        for epoch in range(num_epochs):
            train_loss, train_rmse, train_mae, train_r2 = self._train_one_epoch(train_loader, device, verbose)
            val_loss, val_rmse, val_mae, val_r2 = self._validate_one_epoch(val_loader, device, verbose)

            train_rmse_list.append(train_rmse)
            train_mae_list.append(train_mae)
            train_r2_list.append(train_r2)
            val_rmse_list.append(val_rmse)
            val_mae_list.append(val_mae)
            val_r2_list.append(val_r2)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Train RMSE: {train_rmse_list[-1]:.7f}, MAE: {train_mae_list[-1]:.7f}, R^2: {train_r2_list[-1]:.7f}, "
                      f"Validation RMSE: {val_rmse_list[-1]:.7f}, MAE: {val_mae_list[-1]:.7f}, R^2: {val_r2_list[-1]:.7f}")

            # early stopping logic
            # if val_loss < self.best_val_loss:
            #     self.best_val_loss = val_loss
            #     self.early_stop_counter = 0
            # else:
            #     self.early_stop_counter += 1
            #     if self.early_stop_counter >= self.patience:
            #         print(f"Early stopping triggered. Best validation loss: {self.best_val_loss:.7f}")
            #         break

        min_idx = val_rmse_list.index(min(val_rmse_list))
        return (
            train_rmse_list[min_idx], train_mae_list[min_idx], train_r2_list[min_idx],
            val_rmse_list[min_idx], val_mae_list[min_idx], val_r2_list[min_idx]
        )

    def _train_one_epoch(self, train_loader, device, verbose):
        self.model.train()
        total_loss, total_rmse, total_mae, total_r2 = 0, 0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            mse = torch.mean((outputs - targets) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            mae = torch.mean(torch.abs(outputs - targets)).item()
            r2 = r2_score(targets.cpu().numpy(), outputs.cpu().detach().numpy())

            total_loss += loss.item()
            total_rmse += rmse
            total_mae += mae
            total_r2 += r2

        return (
            total_loss / len(train_loader),
            total_rmse / len(train_loader),
            total_mae / len(train_loader),
            total_r2 / len(train_loader)
        )

    def _validate_one_epoch(self, val_loader, device, verbose):
        self.model.eval()
        total_loss, total_rmse, total_mae, total_r2 = 0, 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                mse = torch.mean((outputs - targets) ** 2).item()
                rmse = torch.sqrt(torch.tensor(mse)).item()
                mae = torch.mean(torch.abs(outputs - targets)).item()
                r2 = r2_score(targets.cpu().numpy(), outputs.cpu().detach().numpy())

                total_loss += loss.item()
                total_rmse += rmse
                total_mae += mae
                total_r2 += r2

        return (
            total_loss / len(val_loader),
            total_rmse / len(val_loader),
            total_mae / len(val_loader),
            total_r2 / len(val_loader)
        )

    def inference(self, dataset, batch_size=16):
        import time
        from torch.utils.data import DataLoader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_time = 0.0
        total_batches = 0

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                start_time = time.time()
                _ = self.model(inputs)
                end_time = time.time()
                total_time += (end_time - start_time)
                total_batches += 1

        avg_inference_time = total_time / total_batches if total_batches > 0 else 0.0
        return avg_inference_time


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        # rmse_loss = torch.sqrt(mse_loss) + 1e-8
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss
