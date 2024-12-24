import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import pandas as pd
import sys
import os
# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建相对路径并添加到 sys.path
tool_dir = os.path.join(parent_dir, 'tools')
sys.path.append(tool_dir)

# 现在可以导入 data_acf 模块了
from data_storage import Data_storage
from data_acf import calc_distances_dict, plot_data_and_acf_compare
from data_preprocessing import data_read_csv, logging_init
from alg_hierarchical_clustering import hierarchical_clustering
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
columns_required = ['machine_id', 'time_stamp', 'cpu_util_percent']

logger = logging_init('../logs/alg_k_means.log')

logger.info("数据初始化中...")
df = data_read_csv(data_csv_path, num_of_rows, container_usage_column_names, columns_required)

# # 确保日期列是 datetime 类型
# df['ds'] = df['time_stamp']
# df['y'] = df['mem_util_percent']
df = df.sort_values(by='time_stamp')

data1 = df[df['machine_id'] == 'm_3129']
data1['y2'] = data1['cpu_util_percent'].max()
data2 = df[df['machine_id'] == 'm_2556']
start1, end1 = data1['time_stamp'].min(), data1['time_stamp'].max()
start2, end2 = data2['time_stamp'].min(), data2['time_stamp'].max()

start = max(start1, start2)
end = min(end1, end2)
data1 = data1[(data1['time_stamp'] >= start) & (data1['time_stamp'] <= end)]
data2 = data2[(data2['time_stamp'] >= start) & (data2['time_stamp'] <= end)]
data1['cpu_util_percent'] = data1['cpu_util_percent'].interpolate(method='linear')
data2['cpu_util_percent'] = data2['cpu_util_percent'].interpolate(method='linear')
# print(data1)
# print(data2)
data_sum = pd.merge(data1, data2, on='time_stamp', how='outer')
data_sum['cpu_util_percent_x'] = data_sum['cpu_util_percent_x'].interpolate(method='linear')
data_sum['cpu_util_percent_y'] = data_sum['cpu_util_percent_y'].interpolate(method='linear')
# print(data_sum)
data_sum['value_sum'] = data_sum['cpu_util_percent_x'] + data_sum['cpu_util_percent_y']
data_sum['y_sum'] = data_sum['value_sum'].max()
# print(data_sum)
# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制 data1 与 time 的关系
# plt.plot(data1['time_stamp'], data1['cpu_util_percent'], label='Data 1', marker='o', linestyle='-', color='b', markersize=3)
# plt.fill_between(data1['time_stamp'], data1['cpu_util_percent'], color='b', alpha=0.4)  # 填充线和x轴之间的面积
# # # 绘制 data2 与 time 的关系
# plt.plot(data1['time_stamp'], data1['y2'], label='Data 2', marker='s', linestyle='--', color='gray', markersize=3)
# plt.fill_between(data1['time_stamp'], data1['y2'], color='gray', alpha=0.4)  # 填充线和x轴之间的面积
# # 绘制 data1 + data2 与 time 的关系
plt.plot(data_sum['time_stamp'], data_sum['value_sum'], label='Data1 + Data2', marker='^', linestyle='-.', color='r', markersize=3)
plt.fill_between(data_sum['time_stamp'], data_sum['value_sum'], color='r', alpha=0.4)  # 填充线和x轴之间的面积
plt.fill_between(data_sum['time_stamp'], data_sum['y_sum'], color='gray', alpha=0.4)  # 填充线和x轴之间的面积
# 添加标题和标签
plt.title('Data1, Data2 and Their Sum Over Time')
plt.xlabel('Time')
plt.ylabel('Value')

# 设置图例
plt.legend()

# 显示图形
plt.xticks(rotation=45)  # 时间标签旋转，使其更易读
plt.tight_layout()  # 调整布局避免标签重叠
plt.savefig('test-Impact-of-volatility-m_3129_m_2556.png')


# # 创建图形和子图布局
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2x2的子图布局
# # 调整间距
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

# # 左边两张小图
# axs[0, 0].plot(data1['time_stamp'], data1['mem_util_percent'], label='Data 1', marker='o', linestyle='-', color='b', markersize=3)
# axs[0, 0].set_title("Plot 1: m_2259")
# axs[0, 0].legend()

# axs[1, 0].plot(data2['time_stamp'], data2['mem_util_percent'], label='Data 2', marker='s', linestyle='--', color='g', markersize=3)
# axs[1, 0].set_title("Plot 2: m_1858")
# axs[1, 0].legend()

# # 右边单独的一张图
# axs[0, 1].plot(data_sum['time_stamp'], data_sum['value_sum'], label='Data1 + Data2', marker='^', linestyle='-.', color='r', markersize=3)
# axs[0, 1].set_title("Plot 3: m_2259_m_1858")
# axs[0, 1].legend()

# # 隐藏下方不需要的子图 (axs[1,1] - 右下角子图)
# axs[1, 1].axis('off')

# plt.savefig('mem-test2.png')
