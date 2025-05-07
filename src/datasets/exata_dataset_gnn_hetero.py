import torch
from torch.utils.data import Dataset


from utils import helpers
from databases.db_manager import DatabaseManager

from src.datasets.data_processor_gnn_homo import DataProcessorGNN
from src._bak.data_processor_gnn_hetero_rat import DataProcessorGNNHeteroRAT


class ExataDatasetGNNHetero(Dataset):
    def __init__(
        self,
        db_name,
        env_filter=None,
        metric="throughput",
        ishetero=False,
    ):
        self.db_name = db_name
        self.db_manager = DatabaseManager()
        self.graphs = []

        if ishetero:
            self.data_processor = DataProcessorGNNHeteroRAT(env_filter=env_filter, metric=metric)
        else:
            self.data_processor = DataProcessorGNN(env_filter=env_filter, metric=metric)

        datasets = self.db_manager.get_all_records(self.db_name)

        data_list = []
        y_values = []
        for data in datasets:
            graph_data = self.data_processor.process_graph(data)
            data_list.append(graph_data)
            y_values.append(graph_data['sta'].y)

        y_values = torch.cat(y_values, dim=0)
        y_mean = y_values.mean()
        y_std = y_values.std()

        if y_std > 0:
            y_values = (y_values - y_mean) / y_std

        idx = 0
        for graph_data in data_list:
            num_targets = graph_data['sta'].y.shape[0]
            graph_data['sta'].y = y_values[idx:idx + num_targets]
            idx += num_targets
            self.graphs.append(graph_data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def inspect_hetero_data(data):
    print("=== Node Information ===")
    for node_type in data.node_types:
        print(f"[Node Type: {node_type}]")
        if 'x' in data[node_type]:
            print(f"  x: {data[node_type].x.shape}")
        if 'y' in data[node_type]:
            print(f"  y: {data[node_type].y.shape}")
        if 'rat_type' in data[node_type]:
            print(f"  rat_type: {data[node_type].rat_type.shape}")

    print("\n=== Edge Information ===")
    for edge_type in data.edge_types:
        print(f"[Edge Type: {edge_type}]")
        if 'edge_index' in data[edge_type]:
            print(f"  edge_index: {data[edge_type].edge_index.shape}")
        if 'edge_attr' in data[edge_type]:
            print(f"  edge_attr: {data[edge_type].edge_attr.shape}")
        if 'rat_type' in data[edge_type]:
            print(f"  rat_type: {data[edge_type].rat_type.shape}")