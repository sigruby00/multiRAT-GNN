import torch
import sys
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset

from utils import helpers
from databases.db_manager import DatabaseManager

# from datasets.data_processor_adv import DataProcessor_adv
from src.datasets.data_processor_gnn_homo import DataProcessorGNN
from datasets.data_processor_gnn_hetero import DataProcessorGNNHetero
from torch_geometric.data import InMemoryDataset
from utils import helpers
import shutil
import os


class ExataDatasetGNNHomo(Dataset):
    def __init__(
        self,
        db_name,
        env_filter=None,
        metric="throughput",
        ishetero=False,
    ):
        self.db_name = db_name
        self.db_manager = DatabaseManager()

        if ishetero:
            self.data_processor = DataProcessorGNNHetero(env_filter=env_filter, metric=metric)
        else:
            self.data_processor = DataProcessorGNN(env_filter=env_filter, metric=metric)

        datasets = self.db_manager.get_all_records(self.db_name)

        self.data_list = []
        y_values = []

        for data in datasets:
            graph_data = self.data_processor.process_graph(data)
            self.data_list.append(graph_data)
            y_values.append(graph_data.y)

        y_values = torch.cat(y_values, dim=0)
        y_mean = y_values.mean()
        y_std = y_values.std()

        print(f"y_mean: {y_mean}, y_std: {y_std}")

        if y_std > 0:
            y_values = (y_values - y_mean) / y_std

        idx = 0
        for graph_data in self.data_list:
            num_targets = graph_data.y.shape[0]
            graph_data.y = y_values[idx:idx + num_targets]
            idx += num_targets

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_param_len(self):
        n_node_feature = self.data_processor.n_node_features
        n_edge_feature = self.data_processor.n_edge_features
        return [n_node_feature, n_edge_feature]