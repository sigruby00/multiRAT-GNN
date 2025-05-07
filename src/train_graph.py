import torch
import sys, os
import pandas as pd
from pathlib import Path
from itertools import product

# Module imports
module_path = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from datasets.exata_dataset_gnn_homo import ExataDatasetGNNHomo
from datasets.exata_dataset_gnn_hetero import ExataDatasetGNNHetero

from trainers.graph_trainer import GraphTrainer
from src.models.graph_model import MPNN, GCNNet, GATNet, HTNetPyG

from utils import paths

class TrainRunner:
    def __init__(self):
        # Define different environment filters for training
        self.env_filter = {
            1: { # Baseline (All Parameters)
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
            2: { # Without Heterogeneity Information
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
            3: { # Without Spatial Information
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
            4: { # Without RSSI
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
            5: { # Without Congestion Information
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

        # Define base combinations, node counts, traffic types, environment types, and metrics
        self.base_comb = [(2, 2)] # wifi, gNB
        self.num_nodes = [20, 50, 100]
        self.traffic_type = ["cbr", "vbr"]
        self.env_type = ["los", "nlos"]
        self.metric = ["throughput", "delay", "jitter"]


    def run_evaluation(self):

        # Generate all parameter combinations for experiments
        param_combinations = product(self.base_comb, self.num_nodes, self.traffic_type, self.env_type, self.metric)

        for (num_wifi, num_gnb), num_node, traffic_type, env_type, metric in param_combinations:

            # Construct simulation database name based on parameters
            self.sim_db = f"w{num_wifi}_g{num_gnb}_n{num_node}_2025"
            if env_type == "nlos":
                self.sim_db += "_nlos"
            else:
                self.sim_db += "_los"

            if traffic_type == "cbr":
                self.sim_db += ""
            else:
                self.sim_db += "_vbr"

            try:
                for filter_key, filter_param in self.env_filter.items():
                    # Load datasets with and without heterogeneity
                    dataset_homo = ExataDatasetGNNHomo(self.sim_db, env_filter=filter_param, metric=metric, ishetero=False)
                    dataset_hetero = ExataDatasetGNNHetero(self.sim_db, env_filter=filter_param, metric=metric, ishetero=True)

                    len_ = dataset_homo.get_param_len()
                    feature_len = {"n_node_features": len_[0], "n_edge_features": len_[1]}

                    for model_type in ["GAT", "ATARI", "MPNN", "HTNetPyG"]:
                        # Initialize models based on type
                        if model_type == "ATARI":
                            model = GCNNet(
                                n_node_features=feature_len["n_node_features"],
                                hidden_dim=64,
                            )
                        elif model_type == "GAT":
                            model = GATNet(
                                n_node_features=feature_len["n_node_features"],
                                hidden_dim=64,
                            )

                        elif model_type == "MPNN":
                            model = MPNN(
                                n_edge_features=feature_len["n_edge_features"],
                                n_node_features=feature_len["n_node_features"],
                                num_hidden=64,
                            )

                        elif model_type == "HTNet":
                            model = HTNetPyG(num_layer=2, dim=64, is_hetero=True,
                                edge_types=[('sta', 'sta_ap', 'ap'), ('sta', 'sta_sta', 'sta')],
                                n_node_features=feature_len["n_node_features"],
                                n_edge_features=feature_len["n_edge_features"])

                        else:
                            raise ValueError(f"Unknown model type: {model_type}")


                        model_path = Path(
                            paths.MODELS_DIR + f"/w{num_wifi}_g{num_gnb}_n{num_node}_f{filter_key}_m{metric}_mm{model_type}_e{env_type}_t{traffic_type}.pt"
                        )


                        # Trainer setup and training
                        trainer = GraphTrainer(model, lr=0.005, writer=None)

                        if model_type == "HTNet" or model_type == "MultiRATGNN" or model_type == "HeteroMetaNetRATAware":
                            _, _, _, val_rmse, val_mae, val_r2, train_time= trainer.train(dataset_hetero, num_node, num_epochs=200, batch_size=32, verbose=True, isHetero=True, k=5)
                        else:
                            _, _, _, val_rmse, val_mae, val_r2, train_time= trainer.train(dataset, num_node, num_epochs=200, batch_size=32, verbose=True, isHetero=False, k=5)


                        # Inference Time Measurement and GPU Memory Usage
                        if model_type in ["HTNet", "MultiRATGNN"]:
                            avg_inference_time = trainer.inference(dataset_hetero, batch_size=1, num_nodes=num_node, isHetero=True)
                            # Insert GPU memory usage print after inference
                            torch.cuda.synchronize()
                            gpu_mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            print(f"[{model_type}] GPU Memory Used (batch=1): {gpu_mem_used:.2f} MB")
                        else:
                            avg_inference_time = trainer.inference(dataset, batch_size=1, num_nodes=num_node, isHetero=False)
                            # Insert GPU memory usage print after inference
                            torch.cuda.synchronize()
                            gpu_mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            print(f"[{model_type}] GPU Memory Used (batch=1): {gpu_mem_used:.2f} MB")

                        # Prepare results
                        result = {
                            "num_wifi": num_wifi,
                            "num_gnb": num_gnb,
                            "num_node": num_node,
                            "train_type": model_type,
                            "metric": metric,
                            "filter": filter_key,
                            "RMSE": val_rmse,
                            "MAE": val_mae,
                            "R^2": val_r2,
                            "los": env_type,
                            "traffic": traffic_type,
                            "train_time": train_time,
                            "inference_time": avg_inference_time,
                        }

                        # Save model
                        trainer.save_model(model_path)

                        # Save results
                        print(f"[{model_type}] w{num_wifi} g{num_gnb} n{num_node} filter:{filter_key} -> RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}, Train: {train_time:.2f}s, Inference: {avg_inference_time:.2f}s")
                        result_df = pd.DataFrame([result])

                        # Save to CSV file
                        result_file = "graph_results.csv"
                        file_exists = os.path.exists(result_file)
                        result_df.to_csv(result_file, mode="a", index=False, header=not file_exists)

            except FileNotFoundError:
                print(f"Database {self.sim_db} not found. Skipping to the next parameter set.")

if __name__ == "__main__":

    train_runner= TrainRunner()
    train_runner.run_evaluation()
