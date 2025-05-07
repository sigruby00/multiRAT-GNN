import pandas as pd
import numpy as np
import torch
import utils.helpers as helpers

class DataProcessorGNNHetero:
    def __init__(self, env_filter, metric):
        self.env_filter = env_filter
        self.metric = metric
        self.mode = 0
        self.node_idx = 0

    def process_graph(self, data):
        from torch_geometric.data import HeteroData

        [UA, comm_type, distance, RSSI, node_SINR, base_SINR, Location, TxPower, Congest_Level] = self.parse_input_data(data)

        if self.metric == "throughput":
            [Throughput] = self.parse_output_data(data)
        elif self.metric == "delay":
            [Throughput] = self.parse_output_data_delay(data)
        elif self.metric == "jitter":
            [Throughput] = self.parse_output_data_jitter(data)

        env_df = self.parse_env_data(data)
        df = env_df

        hetero_data = HeteroData()
        node_features_sta = []
        node_features_ap = []
        edge_index_dict = {
            ('sta', 'sta_ap', 'ap'): [],
            ('sta', 'sta_sta', 'sta'): []
        }
        edge_attr_dict = {
            ('sta', 'sta_ap', 'ap'): [],
            ('sta', 'sta_sta', 'sta'): []
        }
        edge_rat_type_dict = {
            ('sta', 'sta_ap', 'ap'): [],
            ('sta', 'sta_sta', 'sta'): []
        }
        y_sta = []

        # 1. STA 노드 feature 구성
        type_one_hot = [0, 1]  # STA
        for loc, hetero, sinr, congest in zip(Location, comm_type, node_SINR, Congest_Level):
            feat = list(type_one_hot)
            if self.env_filter["Type"]:
                feat += [hetero[0]]
            if self.env_filter["Location"]:
                feat += loc.tolist()
            if self.env_filter["Base_SINR"]:
                feat += [sinr[0]]
            if self.env_filter["Congestion_Level"]:
                feat += [0]
            node_features_sta.append(feat)

        # 2. AP 노드 feature 구성
        type_one_hot = [1, 0]  # AP
        unique_ap_ids = df[df["dst_type"] == "BS"]["dst_id"].unique()
        ap_id_map = {}
        RAT_TYPE_TO_INDEX = {'wifi': 0, 'gnb': 1}
        ap_rat_type_tensor_list = []
        for idx, ap_id in enumerate(unique_ap_ids):
            ap_id_map[ap_id] = idx
            row = df[df["dst_id"] == ap_id].iloc[0]
            bs_loc = row["dst_loc"]
            bs_type = row["src_comm_type"]
            bs_sinr = row["sinr"]
            congest_level = len(df[(df["dst_id"] == ap_id) & (df["connected"] == True)]) / len(node_features_sta)
            feat = list(type_one_hot)
            rat_idx = RAT_TYPE_TO_INDEX[bs_type]
            ap_rat_type_tensor_list.append(rat_idx)
            if self.env_filter["Type"]:
                feat += [rat_idx]
            if self.env_filter["Location"]:
                feat += bs_loc
            if self.env_filter["Base_SINR"]:
                feat += [bs_sinr]
            if self.env_filter["Congestion_Level"]:
                feat += [congest_level]
            node_features_ap.append(feat)

        hetero_data['ap']['rat_type'] = torch.tensor(ap_rat_type_tensor_list, dtype=torch.long)

        # 3. STA-AP edges
        for _, row in df[(df["src_type"] == "STA") & (df["dst_type"] == "BS") & (df["connected"] == True)].iterrows():
            src = row["src_id"]
            dst = ap_id_map[row["dst_id"]]
            rat_type = RAT_TYPE_TO_INDEX[row["src_comm_type"]]
            edge_index_dict[('sta', 'sta_ap', 'ap')].append([src, dst])
            edge_rat_type_dict[('sta', 'sta_ap', 'ap')].append(rat_type)
            feat = [0, 1]  # STA-AP
            if self.env_filter["Distance"]:
                feat += [row["distance"]]
            if self.env_filter["RSSI"]:
                feat += [row["rssi"]]
            if self.env_filter["Base_SINR"]:
                feat += [row["sinr"]]
            edge_attr_dict[('sta', 'sta_ap', 'ap')].append(feat)

        # 4. STA-STA edges
        for _, row in df[(df["src_type"] == "STA") & (df["dst_type"] == "STA")].iterrows():
            src = row["src_id"]
            dst = row["dst_id"]
            edge_index_dict[('sta', 'sta_sta', 'sta')].append([src, dst])
            edge_rat_type_dict[('sta', 'sta_sta', 'sta')].append(-1)  # unknown or N/A
            feat = [1, 0]  # STA-STA
            if self.env_filter["Distance"]:
                feat += [row["distance"]]
            if self.env_filter["RSSI"]:
                feat += [row["rssi"]]
            if self.env_filter["Base_SINR"]:
                feat += [row["sinr"]]
            edge_attr_dict[('sta', 'sta_sta', 'sta')].append(feat)

        # 5. y (STA 노드에만 target 부여)
        for v in Throughput:
            y_sta.append([v.item()])

        # 6. Tensor 변환 및 HeteroData에 저장
        hetero_data['sta'].x = torch.tensor(node_features_sta, dtype=torch.float)
        hetero_data['ap'].x = torch.tensor(node_features_ap, dtype=torch.float)
        hetero_data['sta'].y = torch.tensor(y_sta, dtype=torch.float)

        for rel in edge_index_dict:
            hetero_data[rel].edge_index = torch.tensor(edge_index_dict[rel], dtype=torch.long).t().contiguous()
            hetero_data[rel].edge_attr = torch.tensor(edge_attr_dict[rel], dtype=torch.float)
            hetero_data[rel].rat_type = torch.tensor(edge_rat_type_dict[rel], dtype=torch.long)

        return hetero_data

    # (다음 함수들은 그대로 유지됨)


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
        # print(dfs)

        # NaN 값을 0으로 대체
        dfs["throughput"] = dfs["throughput"].fillna(0)

        # node_idx = 0 -> all nodes/gateways
        if self.node_idx == 0:
            if self.mode == 0:
                # 각 노드의 throughput을 추출
                throughput_per_node = dfs.groupby("ca_id")["throughput"].sum()
                # print("Throughput per Node:")
                # print(throughput_per_node)
                throughput_array = throughput_per_node.to_numpy()
                return (throughput_array.reshape(-1, 1),)

                # ret
            elif self.mode == 1:
                # 각 gateway의 throughput을 추출 (각 게이트웨이에 연결된 노드의 throughput 총합)
                throughput_per_gateway = dfs.groupby("gateway_id")["throughput"].sum()
                # print("Throughput per Gateway:")
                # print(throughput_per_gateway)
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

        # NaN 값을 0으로 대체
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

        # node_idx = 0 -> all nodes/gateways
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