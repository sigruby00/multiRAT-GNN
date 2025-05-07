import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .base_trainer import ModelTrainer
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


class GraphTrainer(ModelTrainer):
    def __init__(self, model, lr=0.01, writer=None, patience=100):
        super().__init__(model, lr, writer)
        self.model = model
        self.criterion = RMSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.writer = writer

        self.patience = patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def inference(self, dataset, batch_size=16, num_nodes=20, isHetero=False):
        import time
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_inference_time = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                if isHetero:
                    batch.y = batch['sta'].y

                mask = torch.zeros_like(batch.y, dtype=torch.bool)
                mask[:num_nodes] = True

                start_time = time.time()
                _ = self.model(batch)
                end_time = time.time()

                total_inference_time += (end_time - start_time)
                total_batches += 1

        avg_inference_time = total_inference_time / total_batches if total_batches > 0 else 0.0
        return avg_inference_time

    def train(self, dataset, num_nodes=20, num_epochs=100, batch_size=16, verbose=False, isHetero=False, train_ratio=0.8, k=5):
        import time
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        train_rmse_all, train_mae_all, train_r2_all = [], [], []
        val_rmse_all, val_mae_all, val_r2_all = [], [], []

        start_time = time.time()
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            # Adjust val_idx size to match train_ratio if needed
            expected_val_size = int((1 - train_ratio) * len(dataset))
            if len(val_idx) > expected_val_size:
                val_idx = val_idx[:expected_val_size]

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            train_rmse_list, train_mae_list, train_r2_list = [], [], []
            val_rmse_list, val_mae_list, val_r2_list = [], [], []

            for epoch in range(num_epochs):
                train_loss, train_rmse, train_mae, train_r2 = self._train_one_epoch(train_loader, num_nodes, device, verbose, isHetero)
                val_loss, val_rmse, val_mae, val_r2 = self._validate_one_epoch(val_loader, num_nodes, device, verbose, isHetero)

                train_rmse_list.append(train_rmse)
                train_mae_list.append(train_mae)
                train_r2_list.append(train_r2)
                val_rmse_list.append(val_rmse)
                val_mae_list.append(val_mae)
                val_r2_list.append(val_r2)

                if (epoch + 1) % 10 == 0:
                    print(f"[Fold {fold+1}/{k}] Epoch {epoch+1}/{num_epochs}, "
                          f"Train RMSE: {train_rmse_list[-1]:.6f}, MAE: {train_mae_list[-1]:.6f}, R^2: {train_r2_list[-1]:.6f}, "
                          f"Validation RMSE: {val_rmse_list[-1]:.6f}, MAE: {val_mae_list[-1]:.6f}, R^2: {val_r2_list[-1]:.6f}")

            min_idx = val_rmse_list.index(min(val_rmse_list))
            train_rmse_all.append(train_rmse_list[min_idx])
            train_mae_all.append(train_mae_list[min_idx])
            train_r2_all.append(train_r2_list[min_idx])
            val_rmse_all.append(val_rmse_list[min_idx])
            val_mae_all.append(val_mae_list[min_idx])
            val_r2_all.append(val_r2_list[min_idx])

        train_time_total = time.time() - start_time
        train_time_avg = train_time_total / k

        return (
            np.mean(train_rmse_all), np.mean(train_mae_all), np.mean(train_r2_all),
            np.mean(val_rmse_all), np.mean(val_mae_all), np.mean(val_r2_all),
            train_time_avg
        )

    def _train_one_epoch(self, data_loader, num_nodes, device, verbose, isHetero=False):
        self.model.train()
        total_loss, total_rmse, total_mae, total_r2 = 0, 0, 0, 0
        total_samples = 0

        for batch in data_loader:
            batch = batch.to(device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            if isHetero:
                node_batch = batch['sta'].batch
                y = batch['sta'].y
            else:
                node_batch = batch.batch
                y = batch.y

            mask = torch.zeros_like(y, dtype=torch.bool)
            node_ptr = 0
            for i in range(batch.num_graphs):
                graph_mask_len = (node_batch == i).sum()
                mask[node_ptr:node_ptr + min(num_nodes, graph_mask_len)] = True
                node_ptr += graph_mask_len

            loss = torch.sqrt(F.mse_loss(out[mask], y[mask]) + 1e-8)
            loss.backward()
            self.optimizer.step()

            mae = torch.mean(torch.abs(out[mask] - y[mask])).item()
            mse = torch.mean((out[mask] - y[mask]) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse, device=device)).item()
            r2 = r2_score(y[mask].cpu().numpy(), out[mask].cpu().detach().numpy())

            total_loss += loss.item() * batch.num_graphs
            total_rmse += rmse * batch.num_graphs
            total_mae += mae * batch.num_graphs
            total_r2 += r2 * batch.num_graphs
            total_samples += batch.num_graphs

        return (
            total_loss / total_samples,
            total_rmse / total_samples,
            total_mae / total_samples,
            total_r2 / total_samples
        )

    def _validate_one_epoch(self, data_loader, num_nodes, device, verbose, isHetero=False):
        self.model.eval()
        total_loss, total_rmse, total_mae, total_r2 = 0, 0, 0, 0
        total_samples = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                out = self.model(batch)
                if isHetero:
                    node_batch = batch['sta'].batch
                    y = batch['sta'].y
                else:
                    node_batch = batch.batch
                    y = batch.y

                mask = torch.zeros_like(y, dtype=torch.bool)
                node_ptr = 0
                for i in range(batch.num_graphs):
                    graph_mask_len = (node_batch == i).sum()
                    mask[node_ptr:node_ptr + min(num_nodes, graph_mask_len)] = True
                    node_ptr += graph_mask_len

                loss = torch.sqrt(F.mse_loss(out[mask], y[mask]) + 1e-8)

                mae = torch.mean(torch.abs(out[mask] - y[mask])).item()
                mse = torch.mean((out[mask] - y[mask]) ** 2).item()
                rmse = torch.sqrt(torch.tensor(mse, device=device)).item()
                r2 = r2_score(y[mask].cpu().numpy(), out[mask].cpu().detach().numpy())

                total_loss += loss.item() * batch.num_graphs
                total_rmse += rmse * batch.num_graphs
                total_mae += mae * batch.num_graphs
                total_r2 += r2 * batch.num_graphs
                total_samples += batch.num_graphs

        return (
            total_loss / total_samples,
            total_rmse / total_samples,
            total_mae / total_samples,
            total_r2 / total_samples
        )


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss
