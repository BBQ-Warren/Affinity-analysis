import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import data_overlap_merge

def data_plot_old_and_merge(data_storage_list, machine_ids_list, interval_index, savefig_name):
    df_list = []
    for data_storage in data_storage_list:
        if data_storage.data_name in machine_ids_list:
            df_list.append(data_storage.interval_df[interval_index])
    # 绘制图形
    plt.figure(figsize=(10, 6))
    print(df_list)
    # 遍历 DataFrame 中除了 'time' 列之外的每一列
    for df in df_list:
        # plt.plot(df['time_stamp'], df['cpu_util_percent'], label=df['machine_id'][0], marker='o')
        print('-------------------------')
        print(df)
        plt.plot(df['time_stamp'], df['cpu_util_percent'], label=df['machine_id'].iloc[0], marker='o')
    new_machine_id = '_'.join(machine_ids_list)
    new_cluster_df = data_overlap_merge(data_storage_list, machine_ids_list, interval_index)
    plt.plot(new_cluster_df['time_stamp'], new_cluster_df['value_sum'], label=new_machine_id, marker='o')
    # 添加标题和标签
    plt.title('Multiple Data Curves Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.xticks(rotation=45)  # 时间标签旋转，使其更易读
    plt.tight_layout()  # 调整布局避免标签重叠
    plt.savefig('./proof/' + savefig_name + '.png')
