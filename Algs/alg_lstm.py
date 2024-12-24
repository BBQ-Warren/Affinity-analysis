import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
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
# 1. 数据加载和预处理
# 假设 'data.csv' 是你的时间序列数据文件，且目标值为 'target'
df = df[df['machine_id'] == 'm_2984']
# 假设数据是单一列时间序列
data = df['cpu_util_percent'].values

# 将数据归一化到 [0, 1] 之间
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# 2. 将数据转换为LSTM需要的格式
# 创建一个函数，将时间序列转换为X（输入数据）和y（目标值）
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

# 选择时间步长（如：1）
time_step = 10  # 例如使用过去10天的数据来预测下一个值
X, y = create_dataset(data_scaled, time_step)

# 将X的形状调整为 LSTM 需要的三维输入：samples, time steps, features
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. 划分训练集和验证集（7:3划分）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, shuffle=False)

# 4. 构建LSTM模型
model = Sequential()

# LSTM层
model.add(LSTM(units=50, return_sequences=False, input_shape=(time_step, 1)))
# 输出层
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 6. 预测结果
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

# 7. 反归一化
train_pred = scaler.inverse_transform(train_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

val_pred = scaler.inverse_transform(val_pred)
y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# 8. 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(y_train_actual, label='True Training Values')
plt.plot(train_pred, label='Predicted Training Values')
plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_val_actual)), y_val_actual, label='True Validation Values')
plt.plot(np.arange(len(train_pred), len(train_pred) + len(val_pred)), val_pred, label='Predicted Validation Values')
plt.legend()
plt.savefig('lstm-test-m_2984.png')

# 计算均方误差（MSE）和均方根误差（RMSE）
train_mse = mean_squared_error(y_train_actual, train_pred)
val_mse = mean_squared_error(y_val_actual, val_pred)

train_rmse = np.sqrt(train_mse)
val_rmse = np.sqrt(val_mse)

# 打印准确度（MSE 和 RMSE）
print(f"Train MSE: {train_mse}")
print(f"Train RMSE: {train_rmse}")
print(f"Validation MSE: {val_mse}")
print(f"Validation RMSE: {val_rmse}")
