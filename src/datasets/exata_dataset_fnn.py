import torch
from scipy.special import erfinv, erf

import numpy as np
import pandas as pd
import random
import math
import sys
from torch.utils.data import Dataset

from utils import helpers
from databases.db_manager import DatabaseManager
from src.datasets.data_processor_fnn import DataProcessor_adv


class ExataDataset_adv(Dataset):

    # node_idx = 0 -> all, 1 -> 1st node, 2 -> 2nd node, ...
    # mode = 0 -> target node 1 -> target gateway
    # env_filter (UAs, Location, RSSI, TX power, Congestion Level)
    def __init__(self, db_name, node_idx=0, mode=0, env_filter=None, metric=None, keep_percentage=1.0):
        if env_filter is None:
            env_filter = {
                "UAs": True,
                "Location": False,
                "RSSI": True,
                "TX_power": False,
                "Congestion_Level": False,
            }
        db_manager = DatabaseManager()
        self.data_processor = DataProcessor_adv(node_idx, mode, env_filter, metric)
        self.data = db_manager.get_all_records(db_name)
        self.node_idx = node_idx
        self.mode = mode
        self.output_min = 0
        self.output_max = 0

        # self.filter_data_by_percentage(keep_percentage)
        self._format_equalizer()

        self.preprocessed_data = [self.data_processor.process_node(data) for data in self.data]
        self.torch_data = [self.preprocess(item) for item in self.preprocessed_data]

        if(metric != "throughput"):
            max_val = self.ret_max_val()
            self.replace_zero_with_max(max_val)

    def replace_zero_with_max(self, max_val):
        replaced_data = []
        for idx in range(self.__len__()):  # 데이터셋의 크기만큼 반복
            input_data, output = self.__getitem__(idx)  # 입력과 출력 데이터를 가져옴
            output = torch.where(output == 0, max_val, output)

            replaced_data.append((input_data, output))

        self.torch_data = replaced_data

    def preprocess(self, item):
        flattened_inputs = torch.flatten(torch.tensor(item[0], dtype=torch.float32))
        flattened_outputs = torch.flatten(torch.tensor(item[1], dtype=torch.float32))

        return flattened_inputs, flattened_outputs


    def ret_max_val(self):
        all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])
        count_greater_than_one = torch.sum(all_outputs > 2).item()
        print(f"Count of values greater than 1: {count_greater_than_one}")
        max_val = torch.max(all_outputs)
        min_val = torch.min(all_outputs)
        mean_val = torch.mean(all_outputs)
        std_val = torch.std(all_outputs)
        print(f"max_val={max_val.item()}, min_val={min_val.item()}, mean_val={mean_val.item()}, std_val={std_val.item()}")

        return max_val.item()

    def normalize_outputs(self):
        """
        Normalize the outputs in torch_data to ensure values are distributed between 0 and 1.
        Combines Z-Score normalization followed by Min-Max normalization.
        """
        # Concatenate all outputs into a single tensor
        all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])

        # Z-Score normalization
        mean = torch.mean(all_outputs)
        std = torch.std(all_outputs)

        print("-------------------------")
        print (mean)
        print (std)
        print("-------------------------")
        zscore_outputs = (all_outputs - mean) / std


        # Normalize each output in torch_data
        scaled_iter = iter(zscore_outputs)  # Iterator for scaled outputs
        self.torch_data = [
            (
                inputs,
                torch.tensor(
                    [next(scaled_iter) for _ in outputs.flatten()]
                ).reshape_as(outputs)
            )
            for inputs, outputs in self.torch_data
        ]

        # Calculate normalized outputs' mean and std
        normalized_all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])
        normalized_mean = torch.mean(normalized_all_outputs)
        normalized_std = torch.std(normalized_all_outputs)

        print(f"Normalized: mean={normalized_mean.item()}, std={normalized_std.item()}")

    def normalize_CDF(self):
        """
        Normalize the outputs in torch_data using Gaussian CDF Transformation.
        Ensures values are distributed between 0 and 1.
        """
        # Concatenate all outputs into a single tensor
        all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])

        # Calculate mean and standard deviation for the Gaussian CDF Transformation
        mean = torch.mean(all_outputs)
        std = torch.std(all_outputs)

        # Gaussian CDF Transformation
        zscore_outputs = ((all_outputs - mean) / std)
        cdf_outputs = 0.5 * (1 + torch.erf(zscore_outputs / math.sqrt(2)))

        # Normalize each output in torch_data
        cdf_iter = iter(cdf_outputs)  # Iterator for CDF-transformed outputs
        self.torch_data = [
            (inputs, torch.tensor([next(cdf_iter) for _ in outputs.flatten()]).reshape_as(outputs))
            for inputs, outputs in self.torch_data
        ]

        # Store normalization parameters for later use
        self.output_mean = mean
        self.output_std = std

        # Calculate normalized outputs' mean and std
        normalized_all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])
        normalized_mean = torch.mean(normalized_all_outputs)
        normalized_std = torch.std(normalized_all_outputs)

        # Print normalization details
        print(f"Original: mean={mean.item()}, std={std.item()}")
        print(f"Normalized: mean={normalized_mean.item()}, std={normalized_std.item()}")

        return mean.item()

    def normalize_MinMax(self):
        """
        Normalize the outputs in torch_data using Min-Max Normalization.
        Ensures values are distributed between 0 and 1.
        """
        # Concatenate all outputs into a single tensor
        all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])

        # Calculate min and max for Min-Max Normalization
        min_val = torch.min(all_outputs)
        max_val = torch.max(all_outputs)

        # Min-Max Normalization
        normalized_outputs = (all_outputs - min_val) / (max_val - min_val)

        # Normalize each output in torch_data
        norm_iter = iter(normalized_outputs)  # Iterator for normalized outputs
        self.torch_data = [
            (inputs, torch.tensor([next(norm_iter) for _ in outputs.flatten()]).reshape_as(outputs))
            for inputs, outputs in self.torch_data
        ]

        # Store normalization parameters for later use
        self.output_min = min_val.item()
        self.output_max = max_val.item()

        # Calculate normalized outputs' min and max
        normalized_all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])
        normalized_min = torch.min(normalized_all_outputs)
        normalized_max = torch.max(normalized_all_outputs)

        # Print normalization details
        print(f"Original: min={min_val.item()}, max={max_val.item()}")
        print(f"Normalized: min={normalized_min.item()}, max={normalized_max.item()}")

        return min_val.item(), max_val.item()

    def denormalize_MinMax(self, normalized_array):
        # Use stored min and max values to denormalize
        normalized_array = np.array(normalized_array)
        original_array = normalized_array * (self.output_max - self.output_min) + self.output_min
        return original_array

    def normalize_MinMax_array(self, original_array):
        # Use stored min and max values to normalize
        original_array = np.array(original_array)
        normalized_array = (original_array - self.output_min) / (self.output_max - self.output_min)
        return normalized_array

    def denormalize_CDF(self, normalized_array):
        normalized_array = np.array(normalized_array)
        all_outputs = np.concatenate([item[1].flatten() for item in self.torch_data])

        # Calculate mean and standard deviation
        mean = np.mean(all_outputs)
        std = np.std(all_outputs)

        # Apply the inverse Gaussian CDF Transformation
        zscore_outputs = np.sqrt(2) * erfinv(2 * normalized_array - 1)
        denormalized_array = zscore_outputs * std + mean


        return denormalized_array

    def normalize_CDF_array(self, array):
        # Concatenate all outputs into a single array
        array = np.array(array)
        all_outputs = np.concatenate([item[1].flatten() for item in self.torch_data])

        # Calculate mean and standard deviation
        mean = np.mean(all_outputs)
        std = np.std(all_outputs)

        # Standardize the input array
        zscore_array = (array - mean) / std

        # Apply the Gaussian CDF Transformation
        normalized_array = 0.5 * (1 + erf(zscore_array / np.sqrt(2)))

        return normalized_array

    def filter_data_by_percentage(self, keep_percentage):
        """
        데이터를 일정 비율만 남기고 나머지를 제거합니다.
        :param keep_percentage: 남길 데이터의 비율 (0.0 ~ 1.0)
        """
        if not (0.0 < keep_percentage <= 1.0):
            raise ValueError("keep_percentage는 0.0과 1.0 사이여야 합니다.")

        # 데이터의 일부만 무작위로 선택하여 남깁니다.
        num_keep = int(len(self.data) * keep_percentage)
        self.data = random.sample(self.data, num_keep)

    def remove_zero_and_below_threshold(self, threshold):
        filtered_data = []
        for data_point in self.torch_data:  # self.data의 각 데이터를 반복
            input_data, output = data_point  # data_point에서 입력과 출력 분리
            # 출력 값에서 0 또는 threshold 이하인 값이 있는지 확인
            if torch.any(output == 0) or torch.any(output > threshold):
                continue  # 조건에 맞으면 건너뛰기
            filtered_data.append(data_point)  # 조건에 맞지 않으면 필터링된 데이터 추가

        # 필터링된 데이터로 업데이트
        self.torch_data = filtered_data

    def remove_outliers_upper_only(self, factor=1.5):
        """
        Remove only upper outliers from self.torch_data based on the IQR method.

        Outliers are defined as values above Q3 + factor * IQR.
        Values below or equal to 0 are always retained.

        Args:
            factor (float): Multiplier for the IQR to define upper outlier thresholds (default is 1.0).
        """
        # Concatenate all outputs into a single tensor for analysis
        all_outputs = torch.cat([item[1].flatten() for item in self.torch_data])

        # Calculate Q3 and IQR
        q3 = torch.quantile(all_outputs, 0.75)
        iqr = q3 - torch.quantile(all_outputs, 0.25)

        # Define the upper outlier threshold
        upper_bound = q3 + factor * iqr

        # Filter out data points with outputs above the upper threshold
        filtered_data = []
        for data_point in self.torch_data:
            input_data, output = data_point
            if torch.any(output > upper_bound):  # Skip if any output exceeds the upper threshold
                continue
            filtered_data.append(data_point)

        # Update the torch_data with filtered data
        self.torch_data = filtered_data

        # Print details of outlier removal
        print(f"Upper outlier threshold: upper={upper_bound.item()}")
        print(f"Original data size: {len(all_outputs)}")
        print(f"Filtered data size: {sum(len(item[1].flatten()) for item in self.torch_data)}")

        return upper_bound.item()

    def remove_zero_data(self):
        filtered_data = []
        for idx in range(self.__len__()):  # 데이터셋의 크기만큼 반복
            _, output = self.__getitem__(idx)  # 입력과 출력 데이터를 가져옴
            zero_indices = torch.where(output == 0)[0]
            if len(zero_indices) == 0:
                filtered_data.append(self.torch_data[idx])

        self.torch_data = filtered_data

    def __len__(self):
        # return len(self.data)
        return len(self.torch_data)

    def get_param_len(self):
        data = self.data[0]
        [inputs, outputs] = self.data_processor.process_node(data)

        # Flatten the inputs and outputs
        flattened_inputs = torch.flatten(torch.tensor(inputs, dtype=torch.float32))
        flattened_outputs = torch.flatten(torch.tensor(outputs, dtype=torch.float32))

        return len(flattened_inputs), len(flattened_outputs)

    def __getitem__(self, idx):
        return self.torch_data[idx]

    def _format_equalizer(self):
        for idx, data in enumerate(self.data):
            for ca in data["data"]["cas"]:
                for conn in ca["connections"]:
                    # RTT와 rssi 값을 float과 int로 변환
                    if "RTT" in conn:
                        conn["RTT"] = float(conn["RTT"])

                    conn["rssi"] = float(conn["rssi"])
                    conn["node_sinr"] = float(conn["node_sinr"])
                    conn["base_sinr"] = float(conn["base_sinr"])

                    # connected 값을 Boolean으로 변환
                    if isinstance(conn["connected"], bool):
                        conn["connected"] = conn["connected"]
                    else:
                        conn["connected"] = True if conn["connected"].lower() == "true" else False

                    # 연결이 안되어 있으면 throughput을 "None"으로 설정
                    if not conn["connected"]:
                        conn["throughput"] = "None"


    def get_numpy_data(self):
        """
        Returns all data in numpy array format for scikit-learn models like XGBoost.
        """
        X = []
        y = []

        for inputs, outputs in self.torch_data:
            X.append(inputs.numpy())
            y.append(outputs.numpy())

        X = np.array(X)
        y = np.array(y)

        # y가 1차원이 되도록 reshape (예: 단일 출력일 때)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()

        return X, y
