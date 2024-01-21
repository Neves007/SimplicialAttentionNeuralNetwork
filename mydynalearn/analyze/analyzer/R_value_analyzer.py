import os
import pandas as pd
import numpy as np

class RValueAnalyzer():
    def __init__(self,config):
        self.config = config
        self.r_value_dataframe_file_path = self.__get_r_value_file_path()
        self.stable_r_value_dataframe_file_path = self.__get_stable_r_value_file_path()
        self.r_value_dataframe = self.__init_r_value_dataframe()
        self.stable_r_value_dataframe = self.__init_stable_r_value_dataframe()

    def __get_r_value_file_path(self):
        root_dir_path = self.config.root_dir_path
        r_value_dir_name = self.config.r_value_dir_name
        # 创建文件夹
        r_value_dir_path = os.path.join(root_dir_path, r_value_dir_name)
        if not os.path.exists(r_value_dir_path):
            os.makedirs(r_value_dir_path)
        # 返回文件路径
        r_value_dataframe_file_name = self.config.r_value_dataframe_file_name
        r_value_dataframe_file_path = os.path.join(r_value_dir_path, r_value_dataframe_file_name)
        return r_value_dataframe_file_path

    def __get_stable_r_value_file_path(self):
        root_dir_path = self.config.root_dir_path
        r_value_dir_name = self.config.r_value_dir_name
        # 创建文件夹
        r_value_dir_path = os.path.join(root_dir_path, r_value_dir_name)
        if not os.path.exists(r_value_dir_path):
            os.makedirs(r_value_dir_path)
        # 返回文件路径
        stable_r_value_dataframe_file_name = self.config.stable_r_value_dataframe_file_name
        stable_r_value_dataframe_file_path = os.path.join(r_value_dir_path, stable_r_value_dataframe_file_name)
        return stable_r_value_dataframe_file_path



    def __init_r_value_dataframe(self):
        r_value_dataframe = pd.DataFrame(columns=["model_network_name",
                                                  "model_dynamics_name",
                                                  "dataset_network_name",
                                                  "dataset_dynamics_name",
                                                  "model_name",
                                                  "model_exp_epoch_index",
                                                  "R"])
        return r_value_dataframe

    def __init_stable_r_value_dataframe(self):
        stable_r_value_dataframe = pd.DataFrame(columns=["model_network_name",
                                                      "model_dynamics_name",
                                                      "dataset_network_name",
                                                      "dataset_dynamics_name",
                                                      "model_name",
                                                      "stable_R_value",
                                                      "max_R_epoch"])
        return stable_r_value_dataframe

    def save_r_value_dataframe(self):
        self.r_value_dataframe.to_csv(self.r_value_dataframe_file_path, index=False)

    def save_stable_r_value_dataframe(self):
        self.stable_r_value_dataframe.to_csv(self.stable_r_value_dataframe_file_path, index=False)

    def load_r_value_dataframe(self):
        self.r_value_dataframe = pd.read_csv(self.r_value_dataframe_file_path)
    def load_stable_r_value_dataframe(self):
        self.stable_r_value_dataframe = pd.read_csv(self.stable_r_value_dataframe_file_path)
    def analyze_result_to_pdseries(self,analyze_result):
        extracted_result = {key: analyze_result[key] for key in self.r_value_dataframe.columns}
        analyze_result_series = pd.Series(extracted_result)
        return analyze_result_series

    def extract_group_properties(self,row):
        columns = ["model_network_name",
                   "model_dynamics_name",
                   "dataset_network_name",
                   "dataset_dynamics_name",
                   "model_name"]
        return {col: row[col] for col in columns}
    def add_r_value(self,analyze_result):
        analyze_result_series = self.analyze_result_to_pdseries(analyze_result)
        self.r_value_dataframe = self.r_value_dataframe.append(analyze_result_series, ignore_index=True)

    def add_first_stable_r_value(self,analyze_result):
        analyze_result_series = self.analyze_result_to_pdseries(analyze_result)
        self.r_value_dataframe = self.r_value_dataframe.append(analyze_result_series, ignore_index=True)


    # 定义一个函数来判断R值是否稳定
    def find_best_epoch(self, r_values, threshold=0.0001, window_size=10):
        '''寻找使 R 值首次稳定的 model_exp_epoch_index

        :param r_values: 按照epoch排序的R值
        :param threshold: 判断稳定的阈值
        :param window_size: 窗口数
        :return: 使 R 值首次稳定的 epoch，始终不稳定返回-1
        '''
        # 计算每个窗口的标准差
        if window_size > len(r_values):
            raise ValueError("Window size cannot be larger than the list of r_values")

        # 遍历r_values找到符合条件的epoch
        for i in range(len(r_values) - window_size):
            current_r = r_values[i]
            # 检查窗口中的R值是否都满足稳定性条件
            window_stable = all(abs(current_r - r) < threshold for r in r_values[i+1:i+window_size])
            if window_stable:
                return i  # 返回第一个稳定的epoch索引
        return -1  # 如果没有找到符合条件的epoch，则返回None
    def get_best_epoch_index(self,
                             model_network_name,
                             model_dynamics_name,
                             dataset_network_name,
                             dataset_dynamics_name,
                             model_name,
                             **kwargs):
        """
        根据提供的参数在stable_r_value_dataframe中查找对应的stable_R_value和max_R_epoch。

        参数:
        - model_network: 模型网络
        - model_dynamics: 模型动力学
        - dataset_network: 数据集网络
        - dataset_dynamics: 数据集动力学
        - model: 模型标识符

        返回:
        - stable_R_value: 稳定的R值
        - max_R_epoch: 达到最大R值的epoch
        """
        # 使用提供的参数在DataFrame中进行查找
        stable_r_value_dataframe = self.stable_r_value_dataframe
        query_result = stable_r_value_dataframe[
            (stable_r_value_dataframe['model_network_name'] == model_network_name) &
            (stable_r_value_dataframe['model_dynamics_name'] == model_dynamics_name) &
            (stable_r_value_dataframe['dataset_network_name'] == dataset_network_name) &
            (stable_r_value_dataframe['dataset_dynamics_name'] == dataset_dynamics_name) &
            (stable_r_value_dataframe['model_name'] == model_name)
            ]

        # 如果有符合条件的行，返回第一条记录的stable_R_value和max_R_epoch
        if not query_result.empty:
            return query_result.iloc[0]['max_R_epoch']
        else:
            # 如果没有找到符合条件的记录，可以返回None或者一些默认值或错误信息
            raise Exception('没有对应的值')
    def analyze_stable_r_value(self):
        self.load_r_value_dataframe()
        if os.path.exists(self.stable_r_value_dataframe_file_path):
            return
        else:
            group_cols = ["model_network_name",
                          "model_dynamics_name",
                          "dataset_network_name",
                          "dataset_dynamics_name",
                          "model_name"]

            # 通过分组获得使R值首次稳定的 model_exp_epoch_index
            grouped = self.r_value_dataframe.groupby(group_cols)

            # 遍历每个分组
            for group_name, group_data in grouped:
                # 获取当前分组的R值列表
                sorted_group_data = group_data.sort_values(by='model_exp_epoch_index')
                r_values = sorted_group_data['R'].values
                # 遍历R值和epoch列表，找到最初进入稳定状态的epoch值
                first_stable_epoch = self.find_best_epoch(r_values)
                properties = self.extract_group_properties(group_data.iloc[0])

                properties["stable_R_value"] = r_values[first_stable_epoch]
                properties["max_R_epoch"] = first_stable_epoch

                self.stable_r_value_dataframe = self.stable_r_value_dataframe.append(properties, ignore_index=True)
            self.save_stable_r_value_dataframe()
        pass