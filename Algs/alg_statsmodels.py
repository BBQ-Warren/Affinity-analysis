import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
import os
from statsmodels.tsa.stattools import acf, pacf
# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建相对路径并添加到 sys.path
tool_dir = os.path.join(parent_dir, 'tools')
sys.path.append(tool_dir)
from data_preprocessing import data_read_csv
from data_acf import plot_data_and_acf_compare, calc_distances_dict
# machine_meta_column_names = ['machine_id', 'time_stamp', 'disaster_level_1', 'disaster_level_2', 'cpu_num', 'mem_size', 'status']
# machine_usage_column_names = ['machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'mpki', 'net_in', 'net_out', 'disk_usage_percent', 'disk_io_percent']
# container_meta_column_names = ['container_id', 'machine_id', 'deploy_unit', 'time_stamp', 'cpu_request', 'cpu_limit', 'mem_size', 'status']
container_usage_column_names = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mpki', 'cpi', 'mem_util_percent', 'mem_gps', 'disk_usage_percent', 'disk_io_percent', 'net_in', 'net_out']
# batch_instance_column_names = ['inst_name', 'task_name', 'task_type', 'job_name', 'status', 'start_time', 'end_time', 'machine_id', 'seq_no', 'total_seq_no', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
# batch_task_column_names = ['task_name', 'inst_num', 'task_type', 'job_name', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem']

dir_path = '../../../alibaba/alibaba_clusterdata_v2018/'
file_name = 'container_usage'
data_csv_path = dir_path + file_name + '.csv'

num_of_rows = 50000
columns_required = ['machine_id', 'time_stamp', 'mem_util_percent']
# 创建一个包含日期（'ds'）和目标值（'y'）的示例数据
df = data_read_csv(data_csv_path, num_of_rows, container_usage_column_names, columns_required)

# 确保日期列是 datetime 类型
df['ds'] = df['time_stamp']
df['y'] = df['mem_util_percent']
df = df.dropna()
distances_dict = calc_distances_dict(df)

# 按照 value 排序字典
sorted_dict = dict(sorted(distances_dict.items(), key=lambda item: (-item[1][0], item[1][1])))

for index, (key, value) in enumerate(sorted_dict.items(), 1):
    print(f"Index {index}: {key} -> {value}")
    plot_data_and_acf_compare(df[df['machine_id'] == key], './mem-acf/mem' + str(index))

