import pandas as pd
import numpy as np
from data_storage import Data_storage
import logging # 日志打印

# csv_name文件路径，num_of_rows:int，column_names和columns_required都是一维列表
def data_read_csv(csv_name, num_of_rows, column_names, columns_required=None):
    df = pd.read_csv(csv_name, nrows=num_of_rows, names=column_names)
    if columns_required is not None:
        df = df[columns_required]
    # df预处理
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='ms')  # 将时间戳转换为 datetime
    df = df.sort_values(by=['machine_id', 'time_stamp'])
    # logger.info(df)
    return df

def data_get_data_storage_list(df):
    data_storage_list = []
    machine_ids = df['machine_id'].unique()
    for machine_id in machine_ids:
        data_df = df[df['machine_id'] == machine_id]
        data_storage = Data_storage(machine_id, data_df)
        data_storage_list.append(data_storage)
    return data_storage_list


# 检查不同实例的时间区间是否有交集
def data_check_overlap(interval_1, interval_2):
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    if start < end:
        return True
    return False

# 更新区间
def data_interval_update(interval_1, interval_2):
    start = max(interval_1[0], interval_2[0])
    end = min(interval_1[1], interval_2[1])
    return [start, end]

# machine_ids是列表
def data_intervals_init(df):
    logger = logging_init('alg_hierarchical_clustering.log')
    # 遍历所有不同的 machine_id 组合
    machine_ids = df['machine_id'].unique()
    len_machine_ids = len(machine_ids)
    logger.info(f"所有的machine个数：{len_machine_ids}", )
    intervals = {}
    # 对于每一个machine_id，取出它的时间区间
    for machine_id in machine_ids:
        intervals[machine_id] = []
        df_machine = df[df['machine_id'] == machine_id]
        start, end = df_machine['time_stamp'].min(), df_machine['time_stamp'].max()
        intervals[machine_id].append(start)
        intervals[machine_id].append(end)
    return intervals

# 根据时间区间不同，分为若干类，同一实例可以在多类
def data_intervals_classification(df, data_storage_list):
    logger = logging_init('alg_hierarchical_clustering.log')
    # # 遍历所有不同的 machine_id 组合
    # machine_ids = df['machine_id'].unique()

    # logger.info('Number of machine_ids:', len(machine_ids))

    logger.info("时间区间初始化...")
    # 初始化时间区间，intervals是字典
    intervals = data_intervals_init(df)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    # Step 2: 划分时间区间组
    intervals_group = []
    intervals_items = []

    for data_storage in data_storage_list:
        add_succ = 0
        for interval_index in range(len(intervals_group)):
            # 如果有交集，那么加入这个组
            interval = intervals_group[interval_index]
            if data_check_overlap(interval, intervals[data_storage.data_name]):
                intervals_group[interval_index] = data_interval_update(interval, intervals[data_storage.data_name])
                intervals_items[interval_index].append(data_storage.data_name)
                add_succ = 1
        if add_succ == 0:
            intervals_group.append(intervals[data_storage.data_name])
            intervals_items.append([data_storage.data_name])


    # for machine_id in machine_ids:
    #     df_machine = df[[df['machine_id'] == machine_id]]

    #     add_succ = 0
    #     for interval_index in range(len(intervals_group)):
    #         # 如果有交集，那么加入这个组
    #         interval = intervals_group[interval_index]
    #         if data_check_overlap(interval, intervals[machine_id]):
    #             intervals_group[interval_index] = data_interval_update(interval, intervals[machine_id])
    #             intervals_items[interval_index].append(machine_id)
    #             add_succ = 1
    #     if add_succ == 0:
    #         intervals_group.append(intervals[machine_id])
    #         intervals_items.append([machine_id])
    len_intervals_group = len(intervals_group)
    logger.info(f'Number of intervals_group:{len_intervals_group}')
    logger.info(intervals_group)
    for intervals_item in intervals_items:
        len_intervals_item = len(intervals_item)
        logger.info(f'Number of intervals_item:{len_intervals_item}')
        logger.info(intervals_item)
    return intervals_group, intervals_items

def data_get_df_through_machine_id(df, machine_id):
    return df[df['machine_id'] == machine_id]

def data_get_df_list_through_machine_ids_and_interval(data_storage_list, machine_ids, interval_index):
    # 取得所有的df
    df_list = []
    for data_storage in data_storage_list:
        if data_storage.data_name in machine_ids:
            df_list.append(data_storage.interval_df[interval_index])
    return df_list

def data_overlap_merge_through_time_stamp(df_list):
    df_machine_i = df_list[0]
    for df_machine_j in df_list[1:]:
        merged_df = pd.merge(df_machine_i, df_machine_j, on='time_stamp', how='outer')
        df_machine_i = merged_df
    return df_machine_i

# def data_get_real_df_through_interval(df_list, interval):
#     for df_index in range(len(df_list)):
#         df_list[df_index] = df_list[df_index][(df_list[df_index]['time_stamp'] >= interval[0]) & (df_list[df_index]['time_stamp'] <= interval[1])]
#     return df_list

def data_get_var(df, cluster_metric = 'cpu_util_percent'):
    return np.var(df[cluster_metric])

def data_get_cv(df, cluster_metric = 'cpu_util_percent', is_merge = False):
    # 计算变异系数
    if is_merge:
        mean_value = df['value_sum'].mean()  # 均值
        std_value = df['value_sum'].std()    # 标准差
    else:
        mean_value = df[cluster_metric].mean()  # 均值
        std_value = df[cluster_metric].std()    # 标准差
    return std_value / mean_value

def data_get_min_volatility_indicator(data_storage_list, machine_ids_list, interval_index, cluster_metric = 'cpu_util_percent', indicator = 'var'):
    logger = logging_init('alg_hierarchical_clustering.log')
    # 取得所有的df
    df_list = data_get_df_list_through_machine_ids_and_interval(data_storage_list, machine_ids_list, interval_index)

    # # 把所有的df根据interval截断
    # df_list = data_get_real_df_through_interval(df_list, interval)

    min_indicator = float('inf')

    if indicator == 'var':
        for df in df_list:
            if data_get_var(df) < min_indicator:
                min_indicator = data_get_var(df)
    elif indicator == 'cv':
        for df in df_list:
            if data_get_cv(df) < min_indicator:
                min_indicator = data_get_cv(df)
    else:
        logger.info('indicator ' + indicator + '不存在')
    return min_indicator

# interval是一个列表，machine_ids_list也是一个列表
def data_overlap_merge(data_storage_list, machine_ids_list, interval_index, cluster_metric = 'cpu_util_percent'):
    # 取得所有的df
    df_list = data_get_df_list_through_machine_ids_and_interval(data_storage_list, machine_ids_list, interval_index)

    # # 把所有的df根据interval截断
    # df_list = data_get_real_df_through_interval(df_list, interval)

    # 把所有的df进行merge
    merged_df = data_overlap_merge_through_time_stamp(df_list)
    merged_df = merged_df.sort_values(by='time_stamp')
    # 找到列名中包含 'cpu_util_percent' 的列
    cpu_cols = [col for col in merged_df.columns if cluster_metric in col]

    # 线性插值
    for cpu_col in cpu_cols:
        merged_df[cpu_col] = merged_df[cpu_col].interpolate(method='linear')

    # 计算这些列的和，并将其存储到 'value_sum' 列
    merged_df['value_sum'] = merged_df[cpu_cols].sum(axis=1)

    # logger.info('merged_df:', merged_df)
    
    return merged_df

def logging_init(log_file_name):
    # 创建日志器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 检查是否已添加处理器
    if not logger.handlers:
        # 创建文件处理器 (FileHandler)
        file_handler = logging.FileHandler(log_file_name, mode='a')  # 写入日志文件
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # 创建控制台处理器 (StreamHandler)
        console_handler = logging.StreamHandler()  # 输出到终端
        console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上日志
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

        # 将处理器添加到日志器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

