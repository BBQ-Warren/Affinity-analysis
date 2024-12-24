# 没理解怎么用
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建相对路径并添加到 sys.path
tool_dir = os.path.join(parent_dir, 'tools')
sys.path.append(tool_dir)
from data_preprocessing import data_read_csv

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
# 创建一个包含日期（'ds'）和目标值（'y'）的示例数据
df = data_read_csv(data_csv_path, num_of_rows, container_usage_column_names, columns_required)

# 确保日期列是 datetime 类型
df['ds'] = df['time_stamp']
df['y'] = df['cpu_util_percent']
df = df[df['machine_id'] == 'm_755']

# 假设 df['y'] 是时间序列数据
time_series = df['y']

machine_ids = df['machine_id'].unique()

data = []
for machine_id in machine_ids:
    data.append(df[df['machine_id'] == machine_id]['y'])

def analyze_periodicity(data, threshold=0.1):
    """
    分析数据的周期性并分类
    :param data: 数据列表，每个元素是一段时间序列
    :param threshold: 判断周期性的功率阈值
    :return: 分为有周期性和无周期性的两类数据
    """
    periodic_data = []
    non_periodic_data = []

    for idx, series in enumerate(data):
        # 进行快速傅里叶变换
        fft_result = np.fft.fft(series)
        power_spectrum = np.abs(fft_result)**2
        frequencies = np.fft.fftfreq(len(series))
        
        # 忽略负频率
        pos_frequencies = frequencies[frequencies > 0]
        pos_power_spectrum = power_spectrum[frequencies > 0]
        
        # 找到主频（最大功率对应的频率）
        max_power = np.max(pos_power_spectrum)
        mean_power = np.mean(pos_power_spectrum)
        print(max_power / mean_power)
        # 判断是否具有显著的周期性
        if max_power / mean_power > threshold:
            periodic_data.append(machine_ids[idx])
        else:
            non_periodic_data.append(machine_ids[idx])

        # # 可视化频谱（可选）
        # plt.figure()
        # plt.plot(pos_frequencies, pos_power_spectrum)
        # plt.title(f'Series {idx + 1} - {"Periodic" if max_power / mean_power > threshold else "Non-Periodic"}')
        # plt.xlabel('Frequency')
        # plt.ylabel('Power Spectrum')
        # plt.savefig('FFT/'+str(machine_ids[idx])+'.png')

    return periodic_data, non_periodic_data
periodic_data, non_periodic_data = analyze_periodicity(data)
print(periodic_data)
print(non_periodic_data)