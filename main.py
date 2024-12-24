from tools.data_preprocessing import data_read_csv, logging_init
from Algs.alg_hierarchical_clustering import hierarchical_clustering
from Algs.alg_statsmodels import
# machine_meta_column_names = ['machine_id', 'time_stamp', 'disaster_level_1', 'disaster_level_2', 'cpu_num', 'mem_size', 'status']
# machine_usage_column_names = ['machine_id', 'time_stamp', 'cpu_util_percent', 'mem_util_percent', 'mpki', 'net_in', 'net_out', 'disk_usage_percent', 'disk_io_percent']
# container_meta_column_names = ['container_id', 'machine_id', 'deploy_unit', 'time_stamp', 'cpu_request', 'cpu_limit', 'mem_size', 'status']
container_usage_column_names = ['container_id', 'machine_id', 'time_stamp', 'cpu_util_percent', 'mpki', 'cpi', 'mem_util_percent', 'mem_gps', 'disk_usage_percent', 'disk_io_percent', 'net_in', 'net_out']
# batch_instance_column_names = ['inst_name', 'task_name', 'task_type', 'job_name', 'status', 'start_time', 'end_time', 'machine_id', 'seq_no', 'total_seq_no', 'cpu_avg', 'cpu_max', 'mem_avg', 'mem_max']
# batch_task_column_names = ['task_name', 'inst_num', 'task_type', 'job_name', 'status', 'start_time', 'end_time', 'plan_cpu', 'plan_mem']

dir_path = '../../alibaba/alibaba_clusterdata_v2018/'
file_name = 'container_usage'
data_csv_path = dir_path + file_name + '.csv'

num_of_rows = 1000000
columns_required = ['machine_id', 'time_stamp', 'cpu_util_percent']

logger = logging_init('alg_hierarchical_clustering.log')


if __name__ == "__main__":
    logger.info("数据初始化中...")
    df = data_read_csv(data_csv_path, num_of_rows, container_usage_column_names, columns_required)

    # 运行层次聚类
    cluster_result, results = hierarchical_clustering(df)

    
    
