import os
from urllib.error import URLError

import pandas as pd

from  load_balance.workload.workload import Workload, WorkloadV2, WorkloadV3
import multiprocessing
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from load_balance.query_balancer_v1 import *
from load_balance.workload.query import Query

class Worker:
    def __init__(self, workload:WorkloadV3,w_type, url, start_time, path, timeout=900):
        self.workload = workload
        self.path = path
        self.start_time = start_time
        self.w_type:str = w_type
        self.setup_sparql(url, timeout)
    
    def setup_sparql(self, url, timeout):
        self.sparql = SPARQLWrapper(url, defaultGraph='http://localhost:8890/dataspace')
        
        self.sparql.setReturnFormat(JSON)
        #sparql.setTimeout(1800)
        self.sparql.setTimeout(timeout)
    
    def execute_query(self, query):
        self.sparql.setQuery(query)
        try:
            ret = self.sparql.query()        
        except TimeoutError:
            return 1
        except Exception as e:
            return e
        return ret

    def execute_query_worker(self):
        workload = self.workload
        
        w_str = self.w_type
        data = []
        #debug code
        #ret = None
        try:
            while True:
                val = workload.FIFO_queue.get()
                if val is None:
                    break
                q = val
                q:Query
                q_start_time = time.perf_counter()
                
                #execute stuff
                ret = self.execute_query(q.query_string)
                q_end_time = time.perf_counter()
                elapsed_time = q_end_time-q_start_time
                try:
                    data.append({
                        'query': str(q), 
                        'start_time':self.start_time, 
                        'arrival_time': q.arrival_time, 
                        'queue_arrival_time':q.queue_arrival_time, 
                        'query_execution_start': q_start_time, 
                        'query_execution_end': q_end_time, 
                        'execution_time': elapsed_time, 
                        'response': ret.reason if isinstance(ret, URLError) else repr(ret) if isinstance(ret, Exception) else 'timed out' if ret == 1 else 'ok'})
                except AttributeError:
                    pass
            with open(f"{self.path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
        except KeyboardInterrupt:
            with open(f"{self.path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
        exit()

def dispatcher(workload: WorkloadV3, start_time, path, n_workers):
    try:
        for numb, (q, a) in enumerate(zip(workload.queries, workload.arrival_times)):
            if numb % 100 == 0:
                s = {}
                s['fifo'] = workload.FIFO_queue.qsize()
                s['time'] = time.perf_counter() - start_time
                print(f"Main process: query {numb} / {len(workload.queries)}: {s}", flush=True)
            n_arr = start_time + a
            q.arrival_time = n_arr
            try:
                time.sleep(n_arr - time.perf_counter())
            except Exception:
                pass
            
            q.queue_arrival_time = time.perf_counter()
            workload.FIFO_queue.put(q)
        
        # Signal stop to workers
        for _ in range(n_workers):
            workload.FIFO_queue.put(None)
        
        with open(f"{path}/main.json", 'w') as f:
            f.write("done")
    except KeyboardInterrupt:
        s = {}
        s['fifo'] = workload.FIFO_queue.qsize()
        s['time'] = time.perf_counter() - start_time
        print(f"Main process: query {numb} / {len(workload.queries)}: {s}")
        with open(f"{path}/main.json", 'w') as f:
            f.write("done")
        exit()           
    exit()


def admission_dispatcher(workload: WorkloadV3, start_time, path, n_workers):
    rej_qs = {'query':[],
              'true_cls': [],
              'time': []}
    try:
        for numb, (q, a) in enumerate(zip(workload.queries, workload.arrival_times)):
            if numb % 100 == 0:
                s = {}
                s['fifo'] = workload.FIFO_queue.qsize()
                s['time'] = time.perf_counter() - start_time
                print(f"Main process: query {numb} / {len(workload.queries)}: {s}", flush=True)
            n_arr = start_time + a
            q.arrival_time = n_arr
            try:
                time.sleep(n_arr - time.perf_counter())
            except Exception:
                pass

            q.queue_arrival_time = time.perf_counter()
            if q.time_cls == 2:
                rej_qs['query'].append(q.ID)
                rej_qs['true_cls'].append(q.true_time_cls)
                rej_qs['time'].append(time.perf_counter())
                continue
            workload.FIFO_queue.put(q)

        # Signal stop to workers
        for _ in range(n_workers):
            workload.FIFO_queue.put(None)

        with open(f"{path}/main.json", 'w') as f:
            f.write("done")

        df = pd.DataFrame(rej_qs)
        df.to_csv(f"{path}/rejected.csv", index=False)
    except KeyboardInterrupt:
        s = {}
        s['fifo'] = workload.FIFO_queue.qsize()
        s['time'] = time.perf_counter() - start_time
        print(f"Main process: query {numb} / {len(workload.queries)}: {s}")
        with open(f"{path}/main.json", 'w') as f:
            f.write("done")
        df = pd.DataFrame(rej_qs)
        df.to_csv(f"{path}/rejected.csv", index=False)
        exit()
    exit()


def main_admission_runner(w, url='http://172.21.233.23:8891/sparql', save_dir='load_balance', n_workers=8):
    f_lb = save_dir
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = {}
    work_names = [f"w_{x + 1}" for x in range(n_workers)]
    start_time = time.perf_counter()
    for work_name in work_names:
        procs[work_name] = multiprocessing.Process(
            target=Worker(w, work_name, url, start_time, path).execute_query_worker)
    procs['main'] = multiprocessing.Process(target=admission_dispatcher, args=(w, start_time, path, n_workers,))

    try:
        for k in procs.keys():
            procs[k].start()

        if w.FIFO_queue.empty():
            for k in work_names:
                procs[k].join()
            procs['main'].join()
        end_time = time.perf_counter()
        print(f"elapsed time: {end_time - start_time}")
        with open(os.path.join(save_dir, 'elapsed_time.txt'), 'w') as f:
            f.write(f"elapsed time: {end_time - start_time}")
    except KeyboardInterrupt:
        end_time = time.perf_counter()
        print(f"elapsed time: {end_time - start_time}")
        with open(os.path.join(save_dir, 'elapsed_time.txt'), 'w') as f:
            f.write(f"elapsed time: {end_time - start_time}")

def main_balance_runner(w, url = 'http://172.21.233.23:8891/sparql', save_dir='load_balance',n_workers=8):
    #f_lb =f'/data/{sample_name}/load_balance_FIFO'
    f_lb = save_dir
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = {}
    work_names = [f"w_{x+1}" for x in range(n_workers)]
    start_time = time.perf_counter()
    for work_name in work_names:
        procs[work_name] = multiprocessing.Process(target=Worker(w,work_name,url, start_time,path).execute_query_worker)
    procs['main'] = multiprocessing.Process(target=dispatcher, args=(w, start_time, path,n_workers,))
    
    try:
        for k in procs.keys():
            procs[k].start()
        
        if w.FIFO_queue.empty():
            for k in work_names:
                procs[k].join()
            procs['main'].join()
        end_time = time.perf_counter()
        print(f"elapsed time: {end_time-start_time}")
    except KeyboardInterrupt:
        end_time = time.perf_counter()
        print(f"elapsed time: {end_time-start_time}")
    

if __name__ == "__main__":
    sample_name="wikidata_0_1_10_v3_path_weight_loss"
    scale="planrgcn_binner_litplan"
    url = "http://172.21.233.14:8891/sparql"
    main_balance_runner(sample_name, scale, url=url)
    # for running for a specific amount of time
    #timeout -s 2 7200 python3 -m load_balance.fifo_balancer
    #for debug
    #timeout -s 2 10 python3 -m load_balance.fifo_balancer
