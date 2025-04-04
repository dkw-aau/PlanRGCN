import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from load_balance.workload.query import Query
import random
from sklearn.model_selection import train_test_split
import multiprocessing
import pickle
class Workload:
    def __init__(self, true_field_name='planrgcn_prediction') -> None:
        self.true_field_name = true_field_name
        self.queries:list[Query] = list()
        self.arrival_times:list[float] = list()
        self.p_idx = 0
        self.current_idx = 0
        self.slow_queue = multiprocessing.Manager().Queue()
        self.med_queue = multiprocessing.Manager().Queue()
        self.fast_queue = multiprocessing.Manager().Queue()
        self.FIFO_queue = multiprocessing.Manager().Queue()
        
    def initialise_queues(self):
        for q, a in zip(self.queries, self.arrival_times):
            match q.time_cls:
                case 0:
                    self.fast_queue.put((q,a))
                case 1:
                    self.med_queue.put((q,a))
                case 2:
                    self.slow_queue.put((q,a))
    
    def initialise_FIFO_que(self):
        for q, a in zip(self.queries, self.arrival_times):
            self.FIFO_queue.put((q,a))
    
    def queries_finished_before_other(self):
        prev, prev_arr, count = None,None, 0
        for q, t in zip(self.queries, self.arrival_times):
            if prev is not None:
                if t > (prev.execution_time + prev_arr):
                    count += 1
            prev = q
            prev_arr=t
        return count
    
    def set_arrival_times(self, arrival_times):
        if not isinstance(arrival_times, list):
            arrival_times = arrival_times.tolist()
        self.arrival_times = arrival_times
    
    def pickle(self, fp):
        pickle.dump((self.queries,self.arrival_times),fp)
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, item):
        if len(self.queries) == len(self.arrival_times):
            return (self.queries[item], self.arrival_times[item])
        return self.queries[item]
    
    def add(self, ID, text, rs):
        q = Query(ID,text,rs)
        self.queries.append(q)
    
    def pop(self):
        if len(self) == 0:
            return None,None
        q = self.queries.pop()
        a = self.arrival_times.pop()
        return q,a
    
    def shuffle_queries(self):
        random.shuffle(self.queries)
        
    def set_time_cls(self, path, add_lsq_url=False):
        df2 = pd.read_csv(path)
        if add_lsq_url:
            df2['id'] = df2['id'].apply(lambda x: f"http://lsq.aksw.org/{x}")
        df2 = df2.set_index('id')
        n_qs = []
        for q in self.queries:
            try:
                q.set_time_cls(df2.loc[q.ID][self.true_field_name])
                q.set_true_time_cls(df2.loc[q.ID]['time_cls'])
                n_qs.append(q)
            except Exception as e:
                print("Workload cannot load for ",q.ID, self.true_field_name)
        print(f"skipped {len(self.queries)-len(n_qs)}")
        self.queries = n_qs
    
    def reorder_queries(self):
        n_queries = len(self.queries)
        time_cls = []
        for q in self.queries:
            match q.true_time_cls:
                case 0:
                    time_cls.append(0)
                case 1:
                    time_cls.append(1)
                case 2:
                    time_cls.append(2)
        queries = []
        X_train = self.queries
        y_train = time_cls
        for _ in range(20):
            X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.001, random_state=42, stratify=y_train)
            queries.extend(X_test)
        
        queries.extend(X_train)
        self.queries = queries
    
    def load_queries(self, path, sep='\t'):
        df = pd.read_csv(path, sep=sep)
        count = 0
        for idx, row in df.iterrows():
            try:
                self.add(row['id'], row['queryString'],row['mean_latency'])
                count += 1
            except Exception as e:
                print(f"loading of query did not work!")
        print(f"loaded {count}")

class WorkloadV2(Workload):
    def __init__(self):
        self.queries:list[Query] = list()
        self.arrival_times:list[float] = list()
        self.p_idx = 0
        self.current_idx = 0
        
        self.queue_dct = {
            "slow" : multiprocessing.Manager().Queue(),
            "med1": multiprocessing.Manager().Queue(),
            "med2": multiprocessing.Manager().Queue(),
            "med3": multiprocessing.Manager().Queue(),
            "fast1": multiprocessing.Manager().Queue(),
            "fast2": multiprocessing.Manager().Queue(),
            "fast3": multiprocessing.Manager().Queue(),
            "fast4": multiprocessing.Manager().Queue(),
        }
        
# For dispatcher
class WorkloadV3(Workload):
    def __init__(self):
        self.queries:list[Query] = list()
        self.arrival_times:list[float] = list()
        self.p_idx = 0
        self.current_idx = 0
        self.slow_queue = multiprocessing.Manager().Queue()
        self.med_queue = multiprocessing.Manager().Queue()
        self.fast_queue = multiprocessing.Manager().Queue()
        self.FIFO_queue = multiprocessing.Manager().Queue()
        
        
        """"self.queue_dct = {
            "slow" : multiprocessing.Manager().Queue(),
            "med1": multiprocessing.Manager().Queue(),
            "med2": multiprocessing.Manager().Queue(),
            "med3": multiprocessing.Manager().Queue(),
            "fast1": multiprocessing.Manager().Queue(),
            "fast2": multiprocessing.Manager().Queue(),
            "fast3": multiprocessing.Manager().Queue(),
            "fast4": multiprocessing.Manager().Queue(),
        }"""

class WorkloadAdmCtrl(Workload):
    def __init__(self, true_field_name='planrgcn_prediction'):
        self.true_field_name = true_field_name
        self.queries: list[Query] = list()
        self.arrival_times: list[float] = list()
        self.p_idx = 0
        self.current_idx = 0
        self.accepted = multiprocessing.Manager().Queue()
        self.rejected = multiprocessing.Manager().Queue()

    def load_workload_file(self, file):
        with open(file, 'rb') as f:
            queries, arrivals = pickle.load(f)
            self.queries = queries
            self.arrival_times = arrivals

class WorkloadAnalyzer:
    def __init__(self, path_to_workload):
        with open(path_to_workload, 'rb') as f:
            self.queries, self.arrival_times = pickle.load(f)

    def plot_act_query_intervals(self):
        intval_ord = []
        pred_ints = []
        for q in self.queries:
            intval_ord.append(q.true_time_cls)
            pred_ints.append( q.time_cls)

        x = []
        for x_1 in range(len(intval_ord)):
            x.append(x_1*2)
        plt.figure(figsize=(10, 6))
        plt.scatter(x, pred_ints, label='Predicted Intervals', marker='o', linestyle='-', color='b')
        # Plot actual runtime intervals
        #plt.scatter(x, intval_ord, label='Actual Intervals', marker='x', linestyle='--', color='r')
        # Add labels and title
        plt.xlabel('Time Index')
        plt.ylabel('Runtime Interval (0: Fast, 1: Medium, 2: Slow)')
        plt.title('Comparison of Predicted and Actual Runtime Intervals')
        plt.legend()
        plt.grid(True)
        plt.show()
