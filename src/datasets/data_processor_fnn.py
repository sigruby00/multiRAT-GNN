import math
import sys
import pandas as pd
import numpy as np

import utils.helpers as helpers


class DataProcessor_adv:
    def __init__(self, node_idx, mode, env_filter, metric):
        self.node_idx = node_idx
        self.mode = mode
        self.env_filter = env_filter
        self.metric = metric

    def process_node_to_df(self):
        pass

    def process_node(self, data):

        # process input data
        [UA, comm_type, distance, RSSI, node_SINR, base_SINR, Location, TxPower, Congest_Level] = self.parse_input_data(data)
        fields = {
            "UAs": UA,
            "Type": comm_type,
            "Distance": distance,
            "RSSI": RSSI,
            "node_SINR": node_SINR,
            "Base_SINR": base_SINR,
            "Location": Location,
            "Tx_power": TxPower,
            "Congestion_Level": Congest_Level,
        }


        input_list = None
        for i, (key, value) in enumerate(self.env_filter.items()):
            if value:
                if input_list is None:  # Initialize input_list with the first True field
                    input_list = fields[key]
                else:  # Stack subsequent fields
                    input_list = np.hstack((input_list, fields[key]))


        # extract input
        if self.node_idx == 0:  # extract all nodes information
            # using filter
            input_vector = input_list
        else:
            # using filter
            input_vector = input_list


        Throughputs = None
        if self.metric == "throughput":
            [Throughputs] = self.parse_output_data(data)
        elif self.metric == "delay":
            [Throughputs] = self.parse_output_data_delay(data)
        elif self.metric == "jitter":
            [Throughputs] = self.parse_output_data_jitter(data)

        # extract output
        if self.node_idx == 0:  # extract all nodes information
            if self.mode == 0:  # extract nodes throughput
                output_vector = Throughputs
            elif self.mode == 1:  # extract gateway throughput
                output_vector = Throughputs
        else:
            if self.mode == 0:  # extract nodes throughput
                output_vector = Throughputs[self.node_idx - 1]
                # output_vector = Throughputs
            elif self.mode == 1:  # extract gateway throughput
                # output_vector = Throughputs
                output_vector = Throughputs[self.node_idx - 1]

        return [input_vector, output_vector]

    def process_node_augment(self, data):

        # process input data
        [UA, comm_type, distance, RSSI, node_SINR, base_SINR, Location, TxPower, Congest_Level] = self.parse_input_data(data)
        fields = {
            "UAs": UA,
            "Type": comm_type,
            "Distance": distance,
            "RSSI": RSSI,
            "node_SINR": node_SINR,
            "Base_SINR": base_SINR,
            "Location": Location,
            "Tx_power": TxPower,
            "Congestion_Level": Congest_Level,
        }

        input_list = None
        for i, (key, value) in enumerate(self.env_filter.items()):
            if value:
                if input_list is None:  # Initialize input_list with the first True field
                    input_list = fields[key]
                else:  # Stack subsequent fields
                    input_list = np.hstack((input_list, fields[key]))

        # extract input
        if self.node_idx == 0:  # extract all nodes information
            input_vector = input_list
        else:
            input_vector = input_list

        Throughputs = None
        if self.metric == "throughput":
            [Throughputs] = self.parse_output_data(data)
        elif self.metric == "delay":
            [Throughputs] = self.parse_output_data_delay(data)
        elif self.metric == "jitter":
            [Throughputs] = self.parse_output_data_jitter(data)

        output_vector = Throughputs

        num_rows = input_vector.shape[0]  # Number of rows in input_vector
        augmented_data = []
        for i in range(num_rows):
            rotated_input = np.roll(input_vector, shift=i, axis=0)  # Rotate input
            rotated_output = np.roll(output_vector, shift=i, axis=0)  # Rotate output
            augmented_data.append([rotated_input, rotated_output])

        return augmented_data


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

        # node_idx = 0 -> all nodes/gateways
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
