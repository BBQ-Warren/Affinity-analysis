import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
# 计算自相关
def calc_acf_value(data_df):
    # 计算自相关函数 (ACF)
    acf_values = acf(data_df['y'], nlags=40)
    return acf_values

def calc_distances_dict(df):
    distances_dict = {}
    machine_ids = df['machine_id'].unique()
    for machine_id in machine_ids:
        data_df = df[df['machine_id'] == machine_id]
        # 计算自相关函数 (ACF)
        acf_values = calc_acf_value(data_df)

        distances_dict[machine_id] = []
        distances_dict[machine_id].append(abs(acf_values[1]))
        distances_dict[machine_id].append(calc_max_point_to_line_distance(acf_values))
    return distances_dict

def plot_data_and_acf_compare(data_df, savefig_name):

    # # 计算偏自相关函数 (PACF)
    # pacf_values = pacf(data_df['y'], nlags=40)
    
    acf_values = calc_acf_value(data_df)
    # 绘制 ACF 和 PACF
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(data_df['ds'], data_df['y'], basefmt=" ", use_line_collection=True)
    plt.title(str(data_df['machine_id'].iloc[0]))

    plt.subplot(1, 2, 2)
    plt.stem(range(len(acf_values)), acf_values, basefmt=" ", use_line_collection=True)
    plt.title('ACF (AutoCorrelation Function)')

    # plt.subplot(1, 3, 3)
    # plt.stem(range(len(pacf_values)), pacf_values, basefmt=" ", use_line_collection=True)
    # plt.title('PACF (Partial AutoCorrelation Function)')

    plt.tight_layout()
    plt.savefig(savefig_name + '.png')

def calc_max_point_to_line_distance(acf_values):
    # 获取第一个点和第40个点
    x1, y1 = 1, acf_values[1]  # 第一个点
    x2, y2 = 40, acf_values[40]  # 第40个点

    # 计算两点之间的直线方程 y = mx + b
    m = (y2 - y1) / (x2 - x1)  # 斜率
    b = y1 - m * x1  # 截距

    distances = [abs(m * x - y + b) / np.sqrt(m**2 + 1) for x, y in enumerate(acf_values[1:])]

    # 找到最大距离
    return max(distances)
