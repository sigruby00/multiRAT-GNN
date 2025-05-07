from sympy import N
import torch
import importlib
import math
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
from utils import paths
import json, csv, re

from simulation import simulation_params


# for cuda
def cuda_available():
    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")


def transform_array(input_array, num_stations, num_gateways):
    """
    Transforms the input array to a tuple indicating which gateway each station has selected.

    Parameters:
    - input_array (tuple): A tuple representing the selection of gateways by stations.
    - num_stations (int): The number of stations.
    - num_gateways (int): The number of gateways.

    Returns:
    - tuple: A tuple where each element represents the selected gateway for each station.

    Example:
    >>> num_stations = 3
    >>> num_gateways = 3
    >>> example1 = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    >>> transform_array(example1, num_stations, num_gateways)
    (-1, -1, -1)

    >>> example2 = (0, 1, 0, 0, 1, 0, 0, 1, 0)
    >>> transform_array(example2, num_stations, num_gateways)
    (1, 1, 1)

    >>> example3 = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    >>> transform_array(example3, num_stations, num_gateways)
    (0, 1, 2)
    """
    if input_array is None:
        raise ValueError("input_array cannot be None")

    if not isinstance(input_array, tuple):
        raise ValueError("input_array must be a tuple")

    if len(input_array) != num_stations * num_gateways:
        raise ValueError("input_array length must be equal to num_stations * num_gateways")

    # Initialize an empty list to store the results.
    result = []

    # Iterate over each station.
    for station in range(num_stations):
        # Iterate over each gateway.
        for gateway in range(num_gateways):
            # Check if the current station has selected the current gateway.
            if input_array[station * num_gateways + gateway] == 1:
                # Add the selected gateway to the result and break the inner loop.
                result.append(gateway)
                break
        else:
            # If no gateway is selected, add -1 to the result.
            result.append(-1)

    # Convert the result list to a tuple and return it.
    return tuple(result)


# id_to_idx, idx_to_id
def ca_id_to_idx(x):
    return x - 1000


def base_id_to_idx(x, num_gnb):
    if x >= 200:
        return x - 200 + num_gnb
    else:
        return x - 100


def base_id_to_base_type(x):
    if x >= 200:
        return "wifi"
    else:
        return "gnb"


def get_avatar_ids(csv_content):
    # Read the CSV content into a DataFrame
    data = pd.read_csv(csv_content, header=None, names=["id", "name"])

    # Ensure the 'name' column is treated as strings
    data["name"] = data["name"].astype(str)

    # Extract the IDs of the Avatars
    avatar_ids = data[data["name"].str.contains("Avatar")]["id"].tolist()
    return avatar_ids


def generate_connection_ids(connection_file, node_id_file, output_file):
    # Read the connection.csv file into a DataFrame
    connection_data = pd.read_csv(connection_file)

    # Read the node_id.csv file into a DataFrame
    node_id_data = pd.read_csv(node_id_file, header=None, names=["id", "name"])
    # print(node_id_data)
    # Merge connection_data with node_id_data to replace names with ids
    merged_data = connection_data.merge(node_id_data, left_on="node", right_on="name", how="left")
    merged_data = merged_data.merge(
        node_id_data,
        left_on="base",
        right_on="name",
        how="left",
        suffixes=("_node", "_base"),
    )

    # print(merged_data)
    # Select only the necessary columns and rename them appropriately
    connection_id_data = merged_data[["id_node", "id_base"]]
    connection_id_data.columns = ["node_id", "base_id"]

    # Save the result to connection_id.csv
    connection_id_data.to_csv(output_file, index=False)

    return 0


# def conver_node_idx_to_id
def convert_id_to_idx(type, id, max_gnb_base):
    if type == "node":
        return id - 1000
    elif type == "base":
        if id >= 200:
            return id - 200 + max_gnb_base
        else:
            return id - 100
    pass


def convert_idx_to_id(type, idx, max_gnb_base):
    if type == "node":
        return idx + 1000
    elif type == "base":
        if idx >= max_gnb_base:
            return idx + 200 - max_gnb_base
        else:
            return idx + 100
    pass


# normalization
def normalized_loc(x, size):
    val = x / size
    return round(val, 2)
    # return val


def normalized_rssi(x):
    val = (x + 100) / 100
    return round(val, 2)
    # return val


def normalized_sinr(x):
    # min_val = 0
    # max_val = 60
    # val = (x - min_val) / (max_val - min_val)
    # return round(val, 2)
    val = (x + 40) / 100
    return round(val, 2)
    # return val


def normalized_tx_power(x):
    return x / 20


def nomalized_throughput(x, offered_load):
    max_throughput = offered_load
    if x == "None":
        return "None"
    else:
        # return round(x / max_throughput, 2)
        return x / max_throughput


# simulation_params.py 파일을 수정하는 함수
def update_terrain_width(new_value):
    # 파일을 읽어와서 내용 수정하기
    file_path = simulation_params.__file__
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 'terrain_width' 라인을 찾아서 값을 변경
    with open(file_path, "w") as file:
        for line in lines:
            if line.startswith("terrain_width"):
                file.write(f"terrain_width = {new_value}\n")

            elif line.startswith("terrain_height"):
                file.write(f"terrain_height = {new_value}\n")
            else:
                file.write(line)

    # 수정 후 모듈 다시 로드
    importlib.reload(simulation_params)


def get_coordinates(node_id, file_path):
    with open(file_path, "r") as file:
        for line in file:
            parts = line.split()
            if parts and parts[0].isdigit() and int(parts[0]) == node_id:
                # 정규 표현식을 사용하여 좌표 추출
                match = re.search(r"\(([^)]+)\)", line)
                if match:
                    coordinates_str = match.group(1)
                    coordinates = coordinates_str.split(",")
                    if len(coordinates) >= 2:
                        try:
                            x = float(coordinates[0].strip())
                            y = float(coordinates[1].strip())
                            return x, y
                        except ValueError:
                            print(f"ValueError for line: {line.strip()}")
                            continue
    return None


def get_base_id_from_csv(filename, node_id):
    with open(filename, mode="r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 건너뛰기
        for row in csv_reader:
            if int(row[0]) == node_id:
                return int(row[1])
    return None  # 해당 node_id가 없을 경우


def get_comm_type(node_id):
    base_id = get_base_id_from_csv(paths.CONNECTIOID_FILE, node_id)
    if base_id >= 200:
        return "wifi"
    else:
        return "gnb"


def get_base_comm_type(base_id):
    if base_id >= 200:
        return "wifi"
    else:
        return "gnb"


def euclidean_distance(src_x, src_y, dest_x, dest_y):
    distance = math.sqrt((dest_x - src_x) ** 2 + (dest_y - src_y) ** 2)
    return distance


def get_node_throughput_session_adv(cursor, table_name, node_id):
    try:
        db_connection = sqlite3.connect(paths.EXATA_DB_FILE)
        # APPLICATION_Summaryを開く
        query = """
            SELECT Timestamp, SenderId, ReceiverId, SessionId, BytesReceived, Throughput, HopCount, Jitter, Delay, ApplicationName, OfferedLoad
            FROM APPLICATION_Summary
            ORDER by Timestamp
        """
        df_summary = pd.read_sql(query, db_connection)  # pd.errors.DatabaseErrorの可能性あり
        time_span = df_summary.iloc[0]["Timestamp"]
        time_span = 1

        if simulation_params.link_type == "uplink":
            df_keys = pd.DataFrame({"SenderId": [node_id], "ReceiverId": [1], "SessionId": [0]})
        elif simulation_params.link_type == "downlink":
            df_keys = pd.DataFrame({"SenderId": [1], "ReceiverId": [node_id], "SessionId": [0]})

        for _, key in df_keys.iterrows():

            df_temp = df_summary[
                (df_summary["SenderId"] == key["SenderId"])
                & (df_summary["ReceiverId"] == key["ReceiverId"])
                # & (df_summary["SessionId"] == key["SessionId"])
                & (df_summary["HopCount"] != 0)
            ]
            df_temp = df_temp.sort_values("Timestamp")  # 念のためソート
            # ----------------------#
            value_arr = df_temp["BytesReceived"].to_numpy()
            value_arr = np.concatenate([[value_arr[0]], value_arr[1:] - value_arr[:-1]])  # erro occurs when value_arr is empty
            # スループット[bps]化
            value_arr = value_arr * 8 / time_span
            throughput = np.average(value_arr[:])
            # ----------------------#

            # throughput = np.average(df_temp["Throughput"].to_numpy())
            delay = np.average(df_temp["Delay"].to_numpy())
            jitter = np.average(df_temp["Jitter"].to_numpy())
            offered_load = np.average(df_temp["OfferedLoad"].to_numpy())

        return throughput, delay, jitter, offered_load

    except Exception as e:
        throughput = 0
        delay = np.NaN
        jitter = np.NaN
        offered_load = 999  # any value (throughput:0) -> coonnection is not eastabilished
        return throughput, delay, jitter, offered_load


def load_wifi_data(db_file):
    try:
        db_connection = sqlite3.connect(db_file)
        query = """
            SELECT EventTimestamp, NodeID, EventName, Pathloss, Sinr, SignalPower, RSNI, Size
            FROM WiFiEvents
            ORDER by EventTimestamp
        """
        df_summary = pd.read_sql(query, db_connection)
        df_summary.dropna(subset=["Pathloss"], inplace=True)  # dropna 적용

        time_start = 11
        time_end = 14
        return df_summary[
            (df_summary["EventTimestamp"] >= time_start) & (df_summary["EventTimestamp"] <= time_end) & (df_summary["EventName"] == "PhyReceived")
        ]
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        # print(f"WiFi 데이터 로드 중 예외 발생: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환


def load_gnb_data(db_file):
    try:
        db_connection = sqlite3.connect(db_file)
        query = """
            SELECT Timestamp, NodeID, EventType, PathLoss, SINR, Size
            FROM PHY_Events
            ORDER by Timestamp
        """
        df_summary = pd.read_sql(query, db_connection)

        time_start = 11
        time_end = 14
        return df_summary[
            (df_summary["Timestamp"] >= time_start) & (df_summary["Timestamp"] <= time_end) & (df_summary["EventType"] == "PhyReceiveSignalAndTB")
        ]
    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        # print(f"gNB 데이터 로드 중 예외 발생: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환    ]

def load_path_loss(db_file):
    try:
        db_connection = sqlite3.connect(db_file)
        query = """
            SELECT X, Y, Z, TxNodeId, PathLoss
            FROM HEATMAP_Events
        """
        pathloss_events= pd.read_sql(query, db_connection)
        return pathloss_events

    except (sqlite3.OperationalError, pd.io.sql.DatabaseError) as e:
        print(f"gNB 데이터 로드 중 예외 발생: {e}")
        return pd.DataFrame()  # 빈 DataFrame 반환    ]



# uplink
def get_node_sinr_session(df_summary, node_id, base_id, path_loss, comm_type=None, time_start=11, time_end=14):

    if comm_type is None:
        if base_id >= 200:
            comm_type = "wifi"
        else:
            comm_type = "gnb"

    path_loss_rounded = round(float(path_loss), 6)
    # if base_id >= 200:
    if comm_type == "wifi":
        # WiFi SINR 계산
        filtered_df = df_summary[(df_summary["NodeID"] == node_id)]
        # filtered_df = df_summary[
        #     (df_summary["NodeID"] == node_id)
        #     & (np.isclose(df_summary["Pathloss"], path_loss_rounded, atol=1e-3))
        # ]
        if filtered_df["Size"].sum() == 0:
            node_sinr_mean = -40
        else:
            node_sinr_mean = (filtered_df["Sinr"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum() if not filtered_df.empty else 0
        # node_signal_power_mean = filtered_df["SignalPower"].mean()
        # node_RSNI_mean = filtered_df["RSNI"].mean()
        # node_sinr_mean = 10 * np.log10((node_signal_power_mean) / (node_RSNI_mean))

        # filtered_df = df_summary[(df_summary["NodeID"] == base_id) & (np.isclose(df_summary["Pathloss"], path_loss_rounded, atol=1e-3))]
        filtered_df = df_summary[(df_summary["NodeID"] == base_id)]
        # filtered_df["Sinr"] = filtered_df["Sinr"].replace("-inf", -20)
        filtered_df.loc[filtered_df["Sinr"] == "-inf", "Sinr"] = -40
        # base_sinr_mean = filtered_df["Sinr"].mean()
        if filtered_df["Size"].sum() == 0:
            base_sinr_mean = -40
        else:
            base_sinr_mean = (filtered_df["Sinr"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum()
        # base_signal_power_mean = filtered_df["SignalPower"].mean()
        # base_RSNI_mean = filtered_df["RSNI"].mean()
        if np.isnan(base_sinr_mean):
            base_sinr_mean = -40

    # elif base_id >= 100:
    elif comm_type == "gnb":
        # gNB SINR 계산
        filtered_df = df_summary[(df_summary["NodeID"] == node_id)]
        if filtered_df["Size"].sum() == 0:
            node_sinr_mean = -40
        else:
            node_sinr_mean = (filtered_df["SINR"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum() if not filtered_df.empty else 0

        # filtered_df = df_summary[(df_summary["NodeID"] == base_id) & (np.isclose(df_summary["PathLoss"], path_loss_rounded, atol=1e-3))]
        filtered_df = df_summary[(df_summary["NodeID"] == base_id)]

        filtered_df.loc[filtered_df["SINR"] == "-inf", "SINR"] = -40
        # base_sinr_mean = filtered_df["SINR"].mean()
        if filtered_df["SINR"].sum() == 0:
            base_sinr_mean = -40
        else:
            base_sinr_mean = (filtered_df["SINR"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum()
        if np.isnan(base_sinr_mean):
            base_sinr_mean = -40

    return node_sinr_mean, base_sinr_mean


def get_node_sinr_session_adv(df_summary, node_id, base_id, path_loss, comm_type=None, time_start=11, time_end=14):

    if comm_type is None:
        if base_id >= 200:
            comm_type = "wifi"
        else:
            comm_type = "gnb"

    if comm_type == "wifi":

        filtered_df = df_summary[(df_summary["NodeID"] == base_id)]
        filtered_df.loc[filtered_df["Sinr"] == "-inf", "Sinr"] = 0
        # base_sinr_mean = filtered_df["Sinr"].mean()

        if filtered_df["Size"].sum() == 0:
            base_sinr_mean = -40
        else:
            base_sinr_mean = (filtered_df["Sinr"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum()

        if np.isnan(base_sinr_mean):
            base_sinr_mean = -40

    # elif base_id >= 100:
    elif comm_type == "gnb":

        filtered_df = df_summary[(df_summary["NodeID"] == base_id)]
        filtered_df.loc[filtered_df["SINR"] == "-inf", "SINR"] = 0
        # base_sinr_mean = filtered_df["SINR"].mean()
        if filtered_df["Size"].sum() == 0:
            base_sinr_mean = -40
        else:
            base_sinr_mean = (filtered_df["SINR"] * filtered_df["Size"]).sum() / filtered_df["Size"].sum()

        if np.isnan(base_sinr_mean):
            base_sinr_mean = -40

    node_sinr_mean = -40

    return node_sinr_mean, base_sinr_mean


def get_node_pathloss(node_id, base_id, comm_type, file_path):
    with open(file_path, "r") as file:
        # Skip the first three lines
        for _ in range(3):
            next(file)

        for line in file:
            columns = line.split()
            # print(f"Number of columns in this line: {len(columns)}")
            if int(columns[0]) == 0 and int(columns[1]) == node_id and int(columns[2]) == base_id:
                if comm_type == "wifi":
                    return float(columns[-1])
                if comm_type == "gnb":
                    return float(columns[3])
                # 3 return columns[3]
                # return columns[4]
    return None

# get node pathloss from heatmap
def get_node_pathloss_heatmap(x, y, base_id, pathloss_df):
    # Round x and y to the nearest integer
    x = round(x)
    y = round(y)
    # Filter the DataFrame to get the pathloss value
    filtered_df = pathloss_df[(pathloss_df['X'] == x) & (pathloss_df['Y'] == y  & (pathloss_df['Z'] == 20)) & (pathloss_df['TxNodeId'] == base_id)]
    if not filtered_df.empty:
        pathloss = filtered_df.iloc[0]['PathLoss']
        return pathloss
    else:
        # return None
        return 150


# Convert numpy.int64 to int if necessary
def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {k: convert_int64_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj


def df_to_json(df):
    cas = []
    for ca_id, group in df.groupby("ca_id"):
        location = {"x": group.iloc[0]["location_x"], "y": group.iloc[0]["location_y"]}
        connections = group.apply(
            lambda row: {
                "rssi": row["rssi"],
                "node_sinr": row["sinr_node"],
                "base_sinr": row["sinr_base"],
                "tx_power": row["tx_power"],
                "connected": row["connected"],
                "gateway_id": row["gateway_id"],
                "throughput": ("None" if pd.isna(row["throughput"]) else row["throughput"]),
            },
            axis=1,
        ).tolist()

        cas.append({"ca_id": ca_id, "location": location, "connections": connections})

    result = {"timestamp": datetime.now().isoformat(), "data": {"cas": cas}}

    # JSON 문자열로 변환

    result = convert_int64_to_int(result)
    json_result = json.dumps(result, indent=4)
    # print(json_result)
    return json_result


def df_to_json_ext(df, df_env):
    cas = []
    for ca_id, group in df.groupby("ca_id"):
        location = {"x": group.iloc[0]["location_x"], "y": group.iloc[0]["location_y"]}
        connections = group.apply(
            lambda row: {
                "rssi": row["rssi"],
                "node_sinr": row["sinr_node"],
                "base_sinr": row["sinr_base"],
                "tx_power": row["tx_power"],
                "connected": row["connected"],
                "gateway_id": row["gateway_id"],
                "throughput": ("None" if pd.isna(row["throughput"]) else row["throughput"]),
                "delay": ("None" if pd.isna(row["delay"]) else row["delay"]),
                "jitter": ("None" if pd.isna(row["jitter"]) else row["jitter"]),
            },
            axis=1,
        ).tolist()

        cas.append({"ca_id": ca_id, "location": location, "connections": connections})

    env = []
    env = df_env.to_json(orient="records", lines=True)
    result = {"timestamp": datetime.now().isoformat(), "data": {"cas": cas, "env": env}}
    # result = {"timestamp": datetime.now().isoformat(), "data": {"cas": cas}}
    json_result = json.dumps(result, indent=4)
    return json_result


#    terminal_id    geohash   latitude   longitude  direction  speed      timestamp base_type  base_id connected       rssi  throughput
# 0            0  xn0jkizby  34.682593  135.186681        3.5    4.3  1722319200000       l5g        0     false -32.063025         NaN
# 1            0  xn0jkizby  34.682593  135.186681        3.5    4.3  1722319200000      wifi        1     false -20.000000         NaN
def df_to_json_nict(df):
    cas = []

    for ca_id, group in df.groupby("terminal_id"):
        general = {
            "update_time": group.iloc[0]["timestamp"],
        }
        location = {
            "geohash": group.iloc[0]["geohash"],
            "lattitude": group.iloc[0]["latitude"],
            "longtitude": group.iloc[0]["longitude"],
        }
        movement = {
            "direction": group.iloc[0]["direction"],
            "speed": group.iloc[0]["speed"],
        }
        connections = group.apply(
            lambda row: {
                "rssi": row["rssi"],
                "connected": row["connected"],
                "gateway_id": row["base_id"],
                "gateway_type": row["base_type"],
                "throughput": ("None" if pd.isna(row["throughput"]) else row["throughput"]),
            },
            axis=1,
        ).tolist()

        cas.append(
            {
                "general": general,
                "ca_id": ca_id,
                "location": location,
                "connections": connections,
            }
        )

    result = {"timestamp": datetime.now().isoformat(), "data": {"cas": cas}}

    # JSON 문자열로 변환

    result = convert_int64_to_int(result)
    json_result = json.dumps(result, indent=4)
    # print(json_result)
    return json_result


def json_to_df(json_str):
    data = json.loads(json_str)
    timestamp = data["timestamp"]
    cas = data["data"]["cas"]

    rows = []
    for ca in cas:
        ca_id = ca["ca_id"]
        location_x = ca["location"]["x"]
        location_y = ca["location"]["y"]
        for conn in ca["connections"]:
            row = {
                "time_stamp": timestamp,
                "ca_id": ca_id,
                "location_x": location_x,
                "location_y": location_y,
                "rssi": conn["rssi"],
                "RTT": conn.get("RTT", None),
                "connected": conn["connected"],
                "gateway_id": conn["gateway_id"],
                "throughput": (conn["throughput"] if conn["throughput"] != "None" else None),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def json_to_df_iab(json_str):
    data = json.loads(json_str)
    timestamp = data["timestamp"]
    cas = data["data"]["cas"]

    rows = []
    for ca in cas:
        ca_id = ca["ca_id"]
        location_x = ca["location"]["x"]
        location_y = ca["location"]["y"]
        direction_t = ca["location"]["t"]
        for conn in ca["connections"]:
            row = {
                "time_stamp": timestamp,
                "ca_id": ca_id,
                "location_x": location_x,
                "location_y": location_y,
                "direction_t": direction_t,
                "rssi": conn["rssi"],
                "RTT": conn.get("RTT", None),
                "connected": conn["connected"],
                "gateway_id": conn["gateway_id"],
                "throughput": (conn["throughput"] if conn["throughput"] != "None" else None),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def json_to_df_ieee(data):
    # data = json.loads(json_str)
    # timestamp = data["timestamp"]
    cas = data["data"]["cas"]

    rows = []
    for ca in cas:
        ca_id = ca["ca_id"]
        location_x = ca["location"]["x"]
        location_y = ca["location"]["y"]
        # direction_t = ca["location"]["t"]
        for conn in ca["connections"]:
            row = {
                # "time_stamp": timestamp,
                "ca_id": ca_id,
                "location_x": location_x,
                "location_y": location_y,
                # "direction_t": direction_t,
                "rssi": conn["rssi"],
                # "RTT": conn.get("RTT", None),
                "connected": conn["connected"],
                "gateway_id": conn["gateway_id"],
                "throughput": (conn["throughput"] if conn["throughput"] != "None" else None),
                "delay": (conn["delay"] if conn["delay"] != "None" else None),
                "jitter": (conn["jitter"] if conn["jitter"] != "None" else None),
                # "delay
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def json_to_df_ieee_sinr(data):
    # data = json.loads(json_str)
    # timestamp = data["timestamp"]
    cas = data["data"]["cas"]

    rows = []
    for ca in cas:
        ca_id = ca["ca_id"]
        location_x = ca["location"]["x"]
        location_y = ca["location"]["y"]
        # direction_t = ca["location"]["t"]
        for conn in ca["connections"]:
            row = {
                # "time_stamp": timestamp,
                "ca_id": ca_id,
                "location_x": location_x,
                "location_y": location_y,
                # "direction_t": direction_t,
                "rssi": conn["rssi"],
                "node_sinr": conn["node_sinr"],
                "base_sinr": conn["base_sinr"],
                # "RTT": conn.get("RTT", None),
                "connected": conn["connected"],
                "gateway_id": conn["gateway_id"],
                "throughput": (conn["throughput"] if conn["throughput"] != "None" else None),
                # "delay
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def realdict_to_format(record):
    # 변환할 데이터에서 필요한 부분 추출
    timestamp = record["timestamp"]
    ca_data = record["data"]["cas"][0]  # 'cas' 리스트에서 첫 번째 항목만 사용
    ca_id = ca_data["ca_id"]
    location = ca_data["location"]
    connections = ca_data["connections"]

    # 'connected'가 true인 gateway들만 추출하고 rssi를 모음
    rssi_values = [conn["rssi"] for conn in connections]

    # 새로운 구조로 변환
    transformed = {
        "timestamp": timestamp,
        "ca_id": ca_id,
        "location": location,
        "rssi_values": rssi_values,
    }

    return transformed


def filter_rssi_threshold(data, num_avatar, threshold=-0.55):
    data_origin = data["rssi"].values.reshape(num_avatar, -1)
    data_filtered = np.copy(data_origin)

    for i in range(data_origin.shape[0]):
        row = data_filtered[i]
        row[row < threshold] = 0  # -0.55보다 작으면 0으로 변경
        if np.all(row == 0):  # 모든 값이 0인 경우
            min_index = np.argmin(data_origin[i])
            data_filtered[i] = 0
            data_filtered[i, min_index] = np.min(data_origin[i])
    data_filtered[data_filtered != 0] = 1

    return data_filtered


# def filter_rssi_threshold(data, num_avatar, threshold=-0.55):
#     data_origin = data["rssi"].values.reshape(num_avatar, -1)
#     data_filtered = np.copy(data_origin)

#     for i in range(data_origin.shape[0]):
#         row = data_filtered[i]
#         row[row < threshold] = 0  # -0.55보다 작으면 0으로 변경
#         if np.all(row == 0):  # 모든 값이 0인 경우
#             min_index = np.argmax(data_origin[i])
#             data_filtered[i] = 0
#             data_filtered[i, min_index] = np.min(data_origin[i])
#     data_filtered[data_filtered != 0] = 1

#     return data_filtered


def filter_rssi_maximum(data, num_avatar):
    data_origin = data["rssi"].values.reshape(num_avatar, -1)
    data_filtered = np.zeros_like(data_origin)

    for i in range(data_origin.shape[0]):
        max_index = np.argmax(data_origin[i])
        data_filtered[i, max_index] = 1

    # return data_filtered

    connections = [np.argmax(row) for row in data_filtered]
    return connections

def filter_rssi_maximum_sqlite(data, num_avatar):

    pass


#      node_id  node_loc_x  node_loc_y  base_id  base_loc_x  base_loc_y base_type       rssi  connected
# 0          0  296.312473   96.569372        0       200.0       200.0      wifi -69.077256      False
# 1          0  296.312473   96.569372        1       800.0       200.0      wifi -80.295206      False
# 2          0  296.312473   96.569372        2       500.0       800.0      wifi -83.366712      False
# 3          0  296.312473   96.569372        3       100.0       100.0       gnb -71.932892      False
# 4          0  296.312473   96.569372        4       300.0       100.0       gnb -40.115311       True
# ..       ...         ...         ...      ...         ...         ...       ...        ...        ...
# 945       49    9.224332  570.964569       14       700.0       500.0       gnb -82.904943      False
# 946       49    9.224332  570.964569       15       100.0       700.0       gnb -70.032921      False
# 947       49    9.224332  570.964569       16       300.0       700.0       gnb -76.124440      False
# 948       49    9.224332  570.964569       17       500.0       700.0       gnb -80.180564      False
# 949       49    9.224332  570.964569       18       700.0       700.0       gnb -83.008305      False
def filter_random(data, num_avatar):
    data_origin = data["rssi"].values.reshape(num_avatar, -1)

    data_filtered = np.zeros_like(data_origin)

    for i in range(data_origin.shape[0]):
        # 해당 행에서 랜덤으로 하나의 인덱스를 선택
        random_index = np.random.choice(data_origin.shape[1])
        data_filtered[i, random_index] = 1

    connections = [np.argmax(row) for row in data_filtered]
    return connections


def filter_random_force(data, reserved, num_avatar, num_wifi, num_gnb):
    print(f"Reserved: {reserved}")
    data_origin = data["rssi"].values.reshape(num_avatar, -1)
    data_filtered = np.zeros_like(data_origin)

    # GNB와 WiFi의 인덱스 설정
    gnb_index_start = 0  # GNB는 0부터 시작
    gnb_index_end = num_gnb  # GNB 끝 인덱스 (exclusive)
    wifi_index_start = num_gnb  # WiFi는 GNB 다음부터 시작
    wifi_index_end = num_gnb + num_wifi  # WiFi 끝 인덱스 (exclusive)

    # 랜덤 선택을 보장하기 위해 아바타 인덱스를 섞는다
    avatar_indices = np.arange(num_avatar)
    np.random.shuffle(avatar_indices)

    # 적어도 reserved 개수만큼 GNB 선택 강제
    gnb_selected_avatars = avatar_indices[:reserved]
    for avatar in gnb_selected_avatars:
        gnb_choice = np.random.choice(range(gnb_index_start, gnb_index_end))
        data_filtered[avatar, gnb_choice] = 1

    # 나머지 아바타들은 WiFi 중에서만 랜덤으로 선택
    remaining_avatars = avatar_indices[reserved:]
    for avatar in remaining_avatars:
        wifi_choice = np.random.choice(range(wifi_index_start, wifi_index_end))
        data_filtered[avatar, wifi_choice] = 1

    # 연결된 게이트웨이 인덱스 반환
    connections = [np.argmax(row) for row in data_filtered]
    print(connections)
    return connections



def filter_capacity(data, num_avatar):
    # Original RSSI data reshaped
    data_origin = data["rssi"].values.reshape(num_avatar, -1)
    data_filtered = np.zeros_like(data_origin)

    # Initialize variables for round-robin
    total_base_stations = data_origin.shape[1]
    current_base_station = 0  # Track the current base station for assignment

    # Loop over each node
    for i in range(data_origin.shape[0]):
        # Assign the current node to the next base station
        data_filtered[i, current_base_station] = 1

        # Update the base station index for the next node
        current_base_station = (current_base_station + 1) % total_base_stations

    # Get the connections as a list of indices
    connections = [np.argmax(row) for row in data_filtered]
    return connections


def filter_semi_random(data, num_avatar):
    # np.random.seed(int(time.time()))
    # Original RSSI data reshaped
    data_origin = data["rssi"].values.reshape(num_avatar, -1)
    data_filtered = np.zeros_like(data_origin)

    # Loop over each node's RSSI values
    # shape[0]은 node의 개수 (행의 개수)
    for i in range(data_origin.shape[0]):
        # Filter the data to get the current node's base stations (wifi and gnb)
        rssi_thresh = -71
        node_data = data[data["node_id"] == i]

        # print(i)
        # Find indices of wifi and gnb base stations for the current node
        wifi_indices = node_data[node_data["base_type"] == "wifi"].index
        gnb_indices = node_data[node_data["base_type"] == "gnb"].index
        # print(gnb_indices)

        wifi_indices = wifi_indices % data_origin.shape[1]
        gnb_indices = gnb_indices % data_origin.shape[1]

        # print(wifi_indices)
        # print(gnb_indices)
        gt_idx = np.random.randint(0, data_origin.shape[1] + 1)
        # print(gt_idx)

        if gt_idx in gnb_indices:
            # print("gnb selected")
            data_filtered[i, gt_idx] = 1
        elif gt_idx in wifi_indices:
            valid_wifi_indices = [idx for idx in wifi_indices if data_origin[i, idx] >= rssi_thresh]
            if valid_wifi_indices:
                # Randomly select one index from valid wifi indices
                selected_idx = np.random.choice(valid_wifi_indices)
                data_filtered[i, selected_idx] = 1
            else:
                selected_idx = np.random.choice(gnb_indices)
                data_filtered[i, selected_idx] = 1
                # print(f"No valid WiFi base stations above threshold for node {i}")
                # continue  # Retry for the current node

            # # data_origin[i, wifi_indices] =
            # # check gt_idx rssi가 -75 이상인지
            # if data_origin[i, gt_idx] >= rssi_thresh:
            #     data_filtered[i, gt_idx] = 1
            # else:
            #     # print(
            #     #     f"RSSI for wifi {gt_idx} is below threshold, retrying for node {i}"
            #     # )
            #     continue  # 현재 i에서 다시 시작

    connections = [np.argmax(row) for row in data_filtered]
    return connections


# Load & save the trained model
def load_model(model, path="model.pth"):
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode if further training is not immediately needed
    print(f"Model loaded from {path}")


def save_model(model, path="model.pth"):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
