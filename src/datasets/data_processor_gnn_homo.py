import math
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import utils.helpers as helpers


class DataProcessorGNN:
    def __init__(self, env_filter, metric):
        self.env_filter = env_filter
        self.metric = metric
        self.mode = 0
        self.node_idx = 0
        self.n_edge_features = 0
        self.n_node_features = 0

    def process_graph(self, data):
        [UA, comm_type, distance, RSSI, node_SINR, base_SINR, Location, TxPower, Congest_Level] = self.parse_input_data(data)

        Throughput = None
        if self.metric == "throughput":
            [Throughput] = self.parse_output_data(data)
        elif self.metric == "delay":
            [Throughput] = self.parse_output_data_delay(data)
        elif self.metric == "jitter":
            [Throughput] = self.parse_output_data_jitter(data)

        env_df = self.parse_env_data(data)

        # Create lists to hold node and edge features
        node_features = []
        edge_index = []
        edge_features = []
        edge_targets = []
        node_mask = []

        features = []

        df = env_df
        type_mapping = {"STA": 0, "BS": 1}  # node type mapping

        num_sta = df["src_id"].nunique()
        num_ap = df[df["dst_type"] == "BS"]["dst_id"].nunique()

        # node feature: type(onehot, sta, ap), type(heterogeneoty), location(nor), sinr(node_sinr), congestion level
        type_one_hot = [0, 1]  # one-hot for station
        for loc, hetero, sinr, congest in zip(Location, comm_type, node_SINR, Congest_Level):
            loc_node = loc.tolist()
            sinr_node = sinr[0]
            hetero_type = hetero[0]
            congest_level = Congest_Level[0]

            features = list(type_one_hot)
            if self.env_filter["Type"]:

                features += [hetero_type]
            if self.env_filter["Location"]:
                features += loc_node
            if self.env_filter["Base_SINR"]:
                features += [sinr_node]
            if self.env_filter["Congestion_Level"]:
                features += [0]

            self.n_node_features = len(features)
            node_features.append(features)

        type_one_hot = [1, 0]  # one-hot for base station
        unique_nodes = df[df["dst_type"] == "BS"]["dst_id"].unique()
        for bs_id in unique_nodes:
            loc_node = df[(df["dst_id"] == bs_id) & (df["dst_type"] == "BS")].iloc[0]["dst_loc"]
            bs_hetero_type = df[(df["dst_id"] == bs_id) & (df["dst_type"] == "BS")]["src_comm_type"].unique()[0]
            bs_sinr = df[(df["dst_id"] == bs_id)  & (df["src_type"] == "STA") & (df["dst_type"] == "BS")]["sinr"].unique()[0]
            congest_level = len(df[(df["dst_id"] == bs_id) & (df["connected"] == True)])/num_sta

            # features = list(type_one_hot) + loc_node + [sinr_node]
            features = list(type_one_hot)
            if self.env_filter["Type"]:
                if bs_hetero_type == "gnb":
                    features += [0]
                elif bs_hetero_type == "wifi":
                    features += [1]
                # features += [bs_hetero_type]
            if self.env_filter["Location"]:
                features += loc_node
            if self.env_filter["Base_SINR"]:
                features += [bs_sinr]
            if self.env_filter["Congestion_Level"]:
                features += [congest_level]
            node_features.append(features)


        # edge feature: type, distance, rssi, sinr
        # edge type (STA-AP, AP-AP)
        edge_type = [0, 1]  # STA-AP
        # filtered_df = df[(df["src_type"] == "STA") & (df["dst_type"] == "BS")]
        filtered_df = df[(df["src_type"] == "STA") & (df["dst_type"] == "BS") & (df["connected"] == True)]
        for index, row in filtered_df.iterrows():
            # add edge index
            src_id = row["src_id"]
            dst_id_mod = row["dst_id"] + num_sta
            edge_index.append([src_id, dst_id_mod])

            # add edge features
            distance = row["distance"]
            rssi = row["rssi"]
            sinr = row["sinr"]
            features = list(edge_type)
            if self.env_filter["Distance"]:
                features += [distance]
            if self.env_filter["RSSI"]:
                features += [rssi]
            if self.env_filter["Base_SINR"]:
                features += [sinr]

            self.n_edge_features= len(features)
            # features = edge_type + [distance] + [rssi] + [sinr]
            edge_features.append(features)

        # print(edge_features)

        edge_type = [1, 0]  # STA-STA
        filtered_df = df[(df["src_type"] == "STA") & (df["dst_type"] == "STA")]
        for index, row in filtered_df.iterrows():
            # add edge index
            src_id = row["src_id"]
            dst_id_mod = row["dst_id"]
            edge_index.append([src_id, dst_id_mod])

            # add edge features
            distance = row["distance"]
            rssi = row["rssi"]
            sinr = row["sinr"]
            features = list(edge_type)
            if self.env_filter["Distance"]:
                features += [distance]
            if self.env_filter["RSSI"]:
                features += [rssi]
            if self.env_filter["Base_SINR"]:
                features += [sinr]
            # features = edge_type + [distance] + [rssi] + [sinr]
            edge_features.append(features)

        # add node targets throughput
        for idx in range(num_sta):  # Mark STA nodes as labeled (1)
            edge_targets.append([Throughput[idx].item()])
            node_mask.append(1)

        for idx in range(num_ap):
            edge_targets.append([0])
            node_mask.append(0)

        # Convert to torch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        y = torch.tensor(edge_targets, dtype=torch.float)  # Throughput as target

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return graph_data

    def normalize(self, tensor):
        """Apply min-max normalization."""
        tensor_min = tensor.min(dim=0, keepdim=True).values
        tensor_max = tensor.max(dim=0, keepdim=True).values
        return (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

    def parse_env_data(self, data):
        env_json = data["data"]["env"]
        df = pd.read_json(pd.io.common.StringIO(env_json), orient="records", lines=True)
        return df


    def parse_input_data(self, data):
        connection_info = []
        throughput_list = []
        txpower_list = []
        rssi_list = []
        node_snr_list = []
        base_snr_list = []
        loc_list = []
        type_list = []

        for node in data["data"]["cas"]:
            # Collect location values
            loc_values = [node["location"]["x"], node["location"]["y"]]
            loc_list.append(loc_values)

            # Collect connection and RSSI values
            rssi_values = []
            node_snr_valuse = []
            base_snr_valuse = []
            for conn in node["connections"]:
                connection_info.append(1 if conn["connected"] else 0)
                if conn["connected"]:
                    txpower_list.append(conn.get("tx_power", 0))
                    throughput = float(conn["throughput"]) if conn["throughput"] not in [None, "None", "none", "NaN"] else 0.0
                    throughput_list.append(throughput)
                    node_snr_valuse.append(conn.get("node_sinr", 0))
                    base_snr_valuse.append(conn.get("base_sinr", 0))
                rssi_values.append(conn.get("rssi", 0))
            rssi_list.append(rssi_values)
            node_snr_list.append(node_snr_valuse)
            base_snr_list.append(base_snr_valuse)

        # Convert lists to numpy arrays
        txpower_array = np.array(txpower_list).reshape(-1, 1) / 20
        loc_array = np.array(loc_list)
        rssi_array = np.array(rssi_list)

        node_sinr_array = np.array(node_snr_list)
        base_sinr_array = np.array(base_snr_list)

        UA_array = np.array(connection_info).reshape(len(data["data"]["cas"]), -1)

        # Calculate congestion level
        congestion = np.sum(UA_array, axis=0)
        indices = [np.where(row == 1)[0][0] for row in UA_array]
        # congestion_level = congestion[indices].reshape(-1, 1) / 20
        congestion_level = congestion[indices].reshape(-1, 1) / len(UA_array)

        env_json = data["data"]["env"]
        df_env = pd.read_json(pd.io.common.StringIO(env_json), orient="records", lines=True)
        df_env = df_env[(df_env["connected"]) & (df_env["src_type"] == "STA") & (df_env["dst_type"] == "BS")]
        comm_type = df_env["src_comm_type"].tolist()
        distance = df_env["distance"].tolist()

        comm_type_ = []
        for item in comm_type:
            if item == "gnb":
                comm_type_.append(0)
            elif item == "wifi":
                comm_type_.append(1)

        comm_type_ = np.array(comm_type_).reshape(-1, 1)
        distance = np.array(distance).reshape(-1, 1)

        return [
            UA_array,
            comm_type_,
            distance,
            rssi_array,
            node_sinr_array,
            base_sinr_array,
            loc_array,
            txpower_array,
            congestion_level,
        ]

    def parse_output_data(self, data):
        dfs = helpers.json_to_df_ieee(data)

        dfs["throughput"] = dfs["throughput"].fillna(0)
        if self.node_idx == 0:
            if self.mode == 0:
                throughput_per_node = dfs.groupby("ca_id")["throughput"].sum()
                throughput_array = throughput_per_node.to_numpy()
                return (throughput_array.reshape(-1, 1),)

            elif self.mode == 1:
                throughput_per_gateway = dfs.groupby("gateway_id")["throughput"].sum()
                throughput_array = throughput_per_gateway.to_numpy()
                return (throughput_array.reshape(-1, 1),)
        else:
            if self.mode == 0:
                node_throughput = dfs.groupby("ca_id")["throughput"].sum()
                throughput_array = node_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)
            if self.mode == 1:
                gateway_throughput = dfs.groupby("gateway_id")["throughput"].sum()
                throughput_array = gateway_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)

    def parse_output_data_delay(self, data):
        dfs = helpers.json_to_df_ieee(data)

        dfs["delay"] = dfs["delay"].astype(float)  # NaN을 유지하며 float으로 변환
        if self.node_idx == 0:
            if self.mode == 0:
                throughput_per_node = dfs.groupby("ca_id")["delay"].sum()
                throughput_array = throughput_per_node.to_numpy()
                return (throughput_array.reshape(-1, 1),)

            elif self.mode == 1:
                throughput_per_gateway = dfs.groupby("gateway_id")["delay"].sum()
                throughput_array = throughput_per_gateway.to_numpy()
                return (throughput_array.reshape(-1, 1),)
        else:
            if self.mode == 0:
                node_throughput = dfs.groupby("ca_id")["delay"].sum()
                throughput_array = node_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)
            if self.mode == 1:
                gateway_throughput = dfs.groupby("gateway_id")["delay"].sum()
                throughput_array = gateway_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)

    def parse_output_data_jitter(self, data):
        dfs = helpers.json_to_df_ieee(data)

        dfs["jitter"] = dfs["jitter"].astype(float)  # NaN을 유지하며 float으로 변환
        if self.node_idx == 0:
            if self.mode == 0:
                throughput_per_node = dfs.groupby("ca_id")["jitter"].sum()
                throughput_array = throughput_per_node.to_numpy()
                return (throughput_array.reshape(-1, 1),)

            elif self.mode == 1:
                throughput_per_gateway = dfs.groupby("gateway_id")["jitter"].sum()
                throughput_array = throughput_per_gateway.to_numpy()
                return (throughput_array.reshape(-1, 1),)
        else:
            if self.mode == 0:
                node_throughput = dfs.groupby("ca_id")["jitter"].sum()
                throughput_array = node_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)
            if self.mode == 1:
                gateway_throughput = dfs.groupby("gateway_id")["jitter"].sum()
                throughput_array = gateway_throughput.to_numpy()
                return (throughput_array.reshape(-1, 1),)
