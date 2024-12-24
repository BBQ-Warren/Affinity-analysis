import matplotlib.pyplot as plt
import numpy as np
from data_storage import Data_storage
from data_preprocessing import data_read_csv, data_intervals_classification, data_overlap_merge, data_get_min_volatility_indicator, data_get_df_list_through_machine_ids_and_interval, data_get_data_storage_list, logging_init, data_get_cv
from data_plot import data_plot_old_and_merge
# 获取组合
from itertools import combinations

def hierarchical_clustering(df):
    logger = logging_init('../logs/alg_k_means.log')
    data_storage_list = data_get_data_storage_list(df)
    # intervals_group是列表，intervals_items是machine_id的列表
    intervals_group, intervals_items = data_intervals_classification(df, data_storage_list)
    
    # 存储data_storage的interval信息
    for interval_index in range(len(intervals_group)):
        interval = intervals_group[interval_index]
        items = intervals_items[interval_index]
        for data_storage in data_storage_list:
            if data_storage.data_name in items:
                data_storage.add_interval(interval, interval_index)

    cluster_result = []

    results = []

    for interval_index in range(len(intervals_group)):
        interval = intervals_group[interval_index]
        items = intervals_items[interval_index]
        logger.info(f"目前正在计算的时间区间：{interval}")
        logger.info(f"其中的成员有：{items}")
        cluster_result_temp = {}
        cluster_num_of_items = 2
        while 1:
            min_distance = float('inf')
            current_total_items = list(combinations(items, cluster_num_of_items))
            for current_items in current_total_items:
                merge_df = data_overlap_merge(data_storage_list, current_items, interval_index)
                
                volatility_indicator = 'cv'
                min_indicator = data_get_min_volatility_indicator(data_storage_list, current_items, interval_index, indicator=volatility_indicator)
                if volatility_indicator == 'var':
                    if np.var(merge_df['value_sum']) < min_indicator:
                        logger.info(f"发现{current_items}具有亲和性")
                        dist = min_indicator / (min_indicator - np.var(merge_df['value_sum']))
                        if dist < min_distance:
                            min_distance = dist
                            merge_pair = current_items
                        else:
                            logger.info(f"但{current_items}不是最优")
                elif volatility_indicator == 'cv':
                    if data_get_cv(merge_df, is_merge=True) < min_indicator:
                        logger.info(f"发现{current_items}具有亲和性")
                        dist = min_indicator / (min_indicator - data_get_cv(merge_df, is_merge=True))
                        if dist < min_distance:
                            min_distance = dist
                            merge_pair = current_items
                        else:
                            logger.info(f"但{current_items}不是最优")
            # 判断是否满足合并条件
            if min_distance < float('inf'):
                df_list = data_get_df_list_through_machine_ids_and_interval(data_storage_list, merge_pair, interval_index)
                machine_id_i = merge_pair[0]
                for machine_id_j in merge_pair[1:]:
                    if machine_id_i in cluster_result_temp:
                        if machine_id_j in cluster_result_temp:
                            cluster_result_temp[machine_id_i].append(cluster_result_temp[machine_id_j])
                        else:
                            cluster_result_temp[machine_id_i].append(machine_id_j)
                    else:
                        if machine_id_j in cluster_result_temp:
                            cluster_result_temp[machine_id_i].append(cluster_result_temp[machine_id_j])
                        else:
                            cluster_result_temp[machine_id_i]=[machine_id_j]

                new_cluster_df = data_overlap_merge(data_storage_list, merge_pair, interval_index)
                new_machine_id = '_'.join(merge_pair)
                new_cluster_df['machine_id'] = new_machine_id
                new_cluster_df['cpu_util_percent'] = new_cluster_df['value_sum']
                data_storage_new = Data_storage(new_machine_id, new_cluster_df)
                data_storage_new.add_interval(interval, interval_index)
                data_storage_new.add_merge(merge_pair)
                data_storage_list.append(data_storage_new)
                items.append(new_machine_id)
                logger.info(f"items 已加入新成员：{new_machine_id}")
                data_plot_old_and_merge(data_storage_list, merge_pair, interval_index, savefig_name=new_machine_id)
                # 方差
                # new_result = {
                #     'interval': interval,
                #     'merge_pair': merge_pair,
                #     'min_var': data_get_min_var(data_storage_list, merge_pair, interval_index),
                #     'merged_variance': np.var(new_cluster_df['cpu_util_percent'])
                # }
                # # 变异系数
                new_result = {
                    'interval': interval,
                    'merge_pair': merge_pair,
                    'min_cv': data_get_min_volatility_indicator(data_storage_list, current_items, interval_index, indicator=volatility_indicator),
                    'merged_cv': data_get_cv(new_cluster_df)
                }
                logger.info(new_result)
                results.append(new_result)
                # items[i] = machine_id_first
                # df[df['machine_id'] == machine_id_first] = new_cluster[['machine_id', 'time_stamp', 'cpu_util_percent']]

                # 删除已合并的簇
                for to_del_item in merge_pair:
                    logger.info(f"items 已删除成员：{items[items.index(to_del_item)]}")
                    del items[items.index(to_del_item)]
            else:
                break
        cluster_result.append(cluster_result_temp)
        # 输出结果
    len_cluster_result = len(cluster_result)
    logger.info(f"最终合并的簇数:{len_cluster_result}")
    logger.info(f"发生合并的簇：{cluster_result}")
    logger.info(f"每次合并的优化过程：{results}")

    return cluster_result, results


