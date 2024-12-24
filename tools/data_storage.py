class Data_storage:
    def __init__(self, data_name, df):
        self.data_name = data_name
        self.df = df
        self.interval_df = {}
        self.merge_items = []
    
    def add_interval(self, interval, interval_index):
        self.interval_df[interval_index] = self.df[(self.df['time_stamp'] >= interval[0]) & (self.df['time_stamp'] <= interval[1])]

    def add_merge(self, merge_pair):
        self.merge_items.append(merge_pair)