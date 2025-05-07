import sys, os
import pandas as pd
import torch
from itertools import product
from torch.utils.data import Subset, DataLoader
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import time

# Module imports
# Add the parent directory to sys.path to allow imports from sibling directories
module_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.datasets.exata_dataset_fnn import ExataDataset_adv
from trainers.offline_trainer import OfflineTrainer
from models.fnn_model import FNNModel_adv

class TrainRunner:
    def __init__(self):
        # Define different environment filters representing various feature subsets for training
        self.env_filter = {
            1: {  # Baseline (All Parameters)
                "UAs": True,
                "Type": True,
                "Location": True,
                "Distance": True,
                "RSSI": True,
                "Node_SINR": False,
                "Base_SINR": True,
                "TX_power": False,
                "Congestion_Level": True,
            },
            2: {  # Without Heterogeneity Information
                "UAs": True,
                "Type": False,
                "Location": True,
                "Distance": True,
                "RSSI": True,
                "Node_SINR": False,
                "Base_SINR": True,
                "TX_power": False,
                "Congestion_Level": True,
            },
            3: {  # Without Spatial Information
                "UAs": True,
                "Type": True,
                "Location": False,
                "Distance": False,
                "RSSI": True,
                "Node_SINR": False,
                "Base_SINR": True,
                "TX_power": False,
                "Congestion_Level": True,
            },
            4: {  # Without RSSI
                "UAs": True,
                "Type": True,
                "Location": True,
                "Distance": True,
                "RSSI": False,
                "Node_SINR": False,
                "Base_SINR": True,
                "TX_power": False,
                "Congestion_Level": True,
            },
            5: {  # Without Congestion Information
                "UAs": True,
                "Type": True,
                "Location": True,
                "Distance": True,
                "RSSI": True,
                "Node_SINR": False,
                "Base_SINR": False,
                "TX_power": False,
                "Congestion_Level": False,
            },
        }

        # Define base combinations of WiFi and gNB counts
        self.base_comb = [(2, 2)]
        # Define number of nodes, traffic types, environment types, and metrics to evaluate
        self.num_nodes = [20, 50, 100]
        self.traffic_type = ["cbr", "vbr"]
        self.env_type = ["nlos", "los"]
        self.metric = ["throughput", "delay", "jitter"]

        # Training hyperparameters
        self.learning_rate = 0.005
        self.hidden_size = [512, 256]
        self.batch_size = 64
        self.epochs = 200
        # Models to train and evaluate
        self.models = ["FNN", "XGBoost"]

    def run_evaluation(self):
        # Generate all combinations of parameters for evaluation
        param_combinations = product(self.base_comb, self.num_nodes, self.traffic_type, self.env_type, self.metric)

        # Iterate over each parameter combination
        for (num_wifi, num_gnb), num_node, traffic_type, env_type, metric in param_combinations:
            # Construct simulation database string based on current parameters
            sim_db = f"w{num_wifi}_g{num_gnb}_n{num_node}_2025"
            sim_db += "_nlos" if env_type == "nlos" else "_los"
            sim_db += "" if traffic_type == "cbr" else "_vbr"

            try:
                # Iterate over each environment filter configuration
                for filter_key, filter_param in self.env_filter.items():
                    # Initialize dataset with current simulation DB, filter, and metric
                    dataset = ExataDataset_adv(sim_db, node_idx=0, mode=0, env_filter=filter_param, metric=metric)
                    # Normalize the dataset outputs for consistent scaling
                    dataset.normalize_outputs()
                    # Get input and output feature lengths
                    len_ = dataset.get_param_len()
                    feature_len = {"input_len": len_[0], "output_len": len_[1]}

                    # Iterate over each model type to train and evaluate
                    for model_type in self.models:
                        # Extract numpy arrays of features and labels from dataset
                        x, y = dataset.get_numpy_data()
                        # Setup 5-fold cross-validation with shuffling and fixed random seed
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)

                        # Lists to store evaluation metrics and timing per fold
                        val_rmse_all, val_mae_all, val_r2_all = [], [], []
                        fold_train_times = []
                        fold_inference_times = []

                        # Perform cross-validation
                        for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
                            # Limit validation set size to 20% of dataset if larger
                            expected_val_size = int(0.2 * len(x))
                            if len(val_idx) > expected_val_size:
                                val_idx = val_idx[:expected_val_size]

                            if model_type == "FNN":
                                # Prepare PyTorch subsets and data loaders for training and validation
                                train_subset = Subset(dataset, train_idx)
                                val_subset = Subset(dataset, val_idx)
                                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
                                val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

                                # Initialize FNN model with specified architecture
                                model = FNNModel_adv(
                                    input_size=feature_len["input_len"],
                                    hidden_sizes=self.hidden_size,
                                    output_size=feature_len["output_len"]
                                )
                                # Initialize trainer with model and learning rate
                                trainer = OfflineTrainer(model, lr=self.learning_rate, writer=None)

                                # Record training start time
                                fold_start = time.time()
                                # Train the model on the current fold without verbose output
                                _, _, _, val_rmse, val_mae, val_r2 = trainer.train(
                                    dataset, ext_train_loader=train_loader, ext_val_loader=val_loader, num_epochs=self.epochs, verbose=False
                                )
                                # Record training duration for the fold
                                fold_train_times.append(time.time() - fold_start)

                                # Measure inference time on the entire dataset with batch size 1
                                inference_start = time.time()
                                _ = trainer.inference(dataset, batch_size=1)
                                fold_inference_times.append(time.time() - inference_start)

                                # Synchronize CUDA to ensure accurate memory measurement
                                torch.cuda.synchronize()
                                # Calculate and print GPU memory used during inference in MB
                                gpu_mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
                                print(f"[FNN] GPU Memory Used (batch=1): {gpu_mem_used:.2f} MB")

                            elif model_type == "XGBoost":
                                # Split data into training and validation sets for XGBoost
                                x_train, y_train = x[train_idx], y[train_idx]
                                x_val, y_val = x[val_idx], y[val_idx]

                                # Record training start time
                                fold_start = time.time()
                                # Initialize XGBoost regressor with GPU acceleration parameters
                                model = XGBRegressor(
                                    tree_method="gpu_hist",
                                    predictor="gpu_predictor",
                                    n_estimators=100,
                                    learning_rate=0.1,
                                    max_depth=6,
                                    objective='reg:squarederror',
                                    verbosity=0
                                )
                                # Train the XGBoost model
                                model.fit(x_train, y_train)
                                # Record training duration for the fold
                                fold_train_times.append(time.time() - fold_start)

                                # Predict on validation set
                                preds = model.predict(x_val)

                                # Calculate evaluation metrics using PyTorch tensors for consistency
                                val_rmse = torch.sqrt(torch.mean((torch.tensor(preds) - torch.tensor(y_val))**2)).item()
                                val_mae = torch.mean(torch.abs(torch.tensor(preds) - torch.tensor(y_val))).item()
                                val_r2 = 1 - torch.sum((torch.tensor(y_val) - torch.tensor(preds)) ** 2) / torch.sum((torch.tensor(y_val) - torch.mean(torch.tensor(y_val))) ** 2)
                                val_r2 = val_r2.item()

                                try:
                                    # Import NVIDIA Management Library for GPU memory measurement
                                    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
                                    nvmlInit()
                                    handle = nvmlDeviceGetHandleByIndex(0)

                                    # Warm up GPU memory by predicting on a small subset
                                    _ = model.predict(x_val[:10])
                                    torch.cuda.synchronize()

                                    # Record GPU memory usage before inference
                                    mem_before = nvmlDeviceGetMemoryInfo(handle).used
                                    # Measure inference time on a single sample
                                    inference_start = time.time()
                                    _ = model.predict(x_val[:1])
                                    fold_inference_times.append(time.time() - inference_start)
                                    torch.cuda.synchronize()
                                    # Record GPU memory usage after inference
                                    mem_after = nvmlDeviceGetMemoryInfo(handle).used

                                    # Calculate and print the difference in GPU memory used during inference in MB
                                    gpu_mem_used = (mem_after - mem_before) / (1024 ** 2)
                                    print(f"[XGBoost] GPU Memory Used (batch=1): {gpu_mem_used:.2f} MB")

                                    nvmlShutdown()
                                except ImportError:
                                    # If pynvml is not installed, skip GPU memory usage measurement
                                    print("[XGBoost] pynvml not installed, skipping GPU memory usage print.")

                            # Append metrics from current fold to lists for averaging later
                            val_rmse_all.append(val_rmse)
                            val_mae_all.append(val_mae)
                            val_r2_all.append(val_r2)

                        # Compute average metrics and timing over all folds
                        avg_rmse = np.mean(val_rmse_all)
                        avg_mae = np.mean(val_mae_all)
                        avg_r2 = np.mean(val_r2_all)
                        avg_train_time = np.mean(fold_train_times) if fold_train_times else 0.0
                        avg_inference_time = np.mean(fold_inference_times) if fold_inference_times else 0.0

                        # Prepare result dictionary for logging
                        result = {
                            "num_wifi": num_wifi,
                            "num_gnb": num_gnb,
                            "num_node": num_node,
                            "train_type": model_type,
                            "metric": metric,
                            "filter": filter_key,
                            "RMSE": avg_rmse,
                            "MAE": avg_mae,
                            "R^2": avg_r2,
                            "los": env_type,
                            "traffic": traffic_type,
                            "train_time": avg_train_time,
                            "inference_time": avg_inference_time,
                        }

                        # Print summary of evaluation results for current parameter set and model
                        print(f"[{model_type}] {sim_db} | RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R^2: {avg_r2:.4f}, Train Time: {avg_train_time:.2f}s, Inference Time: {avg_inference_time:.6f}s")

                        # Save results to CSV file, appending if file exists, otherwise creating with header
                        df = pd.DataFrame([result])
                        result_file = "non_graph_results.csv"
                        df.to_csv(result_file, mode="a", index=False, header=not os.path.exists(result_file))

            except FileNotFoundError:
                # Handle missing simulation database files gracefully by skipping
                print(f"Database {sim_db} not found. Skipping.")

if __name__ == "__main__":
    # Instantiate and run the training and evaluation runner
    runner = TrainRunner()
    runner.run_evaluation()
