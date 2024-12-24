import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import os

# 获取当前文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 构建相对路径并添加到 sys.path
tool_dir = os.path.join(parent_dir, 'tools')
sys.path.append(tool_dir)

# 现在可以导入 data_acf 模块了
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
columns_required = ['machine_id', 'time_stamp', 'mem_util_percent']

logger = logging_init('../logs/alg_k_means.log')

logger.info("数据初始化中...")
df = data_read_csv(data_csv_path, num_of_rows, container_usage_column_names, columns_required)

# 确保日期列是 datetime 类型
df['ds'] = df['time_stamp']
df['y'] = df['mem_util_percent']
df = df.dropna()
distances_dict = calc_distances_dict(df)

# 按照 value 排序字典
sorted_dict = dict(sorted(distances_dict.items(), key=lambda item: (-item[1][0], item[1][1])))

X = []


for index, (key, value) in enumerate(sorted_dict.items(), 1):
    # print(f"Index {index}: {key} -> {value}")
    # plot_data_and_acf_compare(df[df['machine_id'] == key], str(index))
    if index == 1:
        the_best_mechine_id = key
        the_best_value = value
    X.append(value)

X = np.array(X)
print(X)
X = X[~np.isnan(X).any(axis=1)]
print(X)
# # 可视化数据
# plt.scatter(X[:, 0], X[:, 1])
# plt.title("Generated Data")
# plt.show()


# 创建 KMeans 模型，将数据分成 2 类
kmeans = KMeans(n_clusters=2, random_state=0)

# 拟合模型
kmeans.fit(X)

# 获取聚类结果的标签（即每个数据点所属的类别）
labels = kmeans.labels_

# 查找value对应的key
def find_key_by_value(dictionary, target_value):
    keys = [key for key, value in dictionary.items() if np.any(value == target_value)]  # 检查数组中是否有任何元素等于target_value
    return keys

the_best_mechine_ids = []


# 打印每个类别包含的值
for cluster_num in np.unique(labels):
    print(f"Cluster {cluster_num}:")
    # 获取当前类的索引
    cluster_values = X[labels == cluster_num]
    if the_best_value in cluster_values:
        the_best_cluster = cluster_num
    for i in cluster_values:
        machine_id = find_key_by_value(distances_dict, i)
        print(machine_id)  # 展开打印
        if the_best_value in cluster_values:
            if the_best_cluster == cluster_num:
                the_best_mechine_ids.append(machine_id[0])
        # 画图
        # plot_data_and_acf_compare(df[df['machine_id'] == machine_id[0]], str(cluster_num) + '/' + str(machine_id[0]) )
    


# 获取聚类中心
centroids = kmeans.cluster_centers_

# # 可视化聚类结果
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')

# # 绘制聚类中心
# plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
# plt.title("KMeans Clustering (k=2)")
# plt.legend()
# plt.savefig('kmeans-test.png')






# print(the_best_mechine_ids)
# df = df[df['machine_id'].isin(the_best_mechine_ids)]

# hierarchical_clustering(df)
