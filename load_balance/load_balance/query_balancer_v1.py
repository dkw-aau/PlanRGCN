import os
import load_balance.workload.arrival_time
from load_balance.workload.arrival_time import ArrivalRateDecider
import pandas as pd
from  load_balance.workload.workload import Workload, WorkloadV2
import copy
import multiprocessing
import time, datetime
from SPARQLWrapper import SPARQLWrapper, JSON
import random
import numpy as np
import json


def execute_query(query,sparql, timeout=1800):
    
    sparql.setQuery(query)
    try:
        ret = sparql.query()        
    except Exception as e:
        return None
    return ret

def monitor_queue(workload: WorkloadV2, start_time, path):
    data = []
    while True:
        s = {}
        nonempty = 0
        for k in workload.queue_dct.keys():
            s[k] = workload.queue_dct[k].qsize()
            if s[k] > 0:
                nonempty=1
        s['time'] = time.perf_counter()
        data.append(s)
        if len(data) % 3 == 0:
            with open(f"{path}/monitor_con.json", 'w') as f:
                json.dump(data, f)
        if nonempty == 0 and os.path.exists(f"{path}/main.json"):
            with open(f"{path}/monitor.json", 'w') as f:
                json.dump(data, f)
            exit()
        print(f"Monitor process: {time.perf_counter()-start_time} s")
        time.sleep(10)
    
def main_worker(workload: WorkloadV2, start_time, path):
    fast_keys =  ["fast1", "fast2", "fast3", "fast4"]
    fast_idx = 0
    medium_keys = ["med1", "med2", "med3"]
    med_idx = 0
    for numb, (q, a) in enumerate(zip(workload.queries, workload.arrival_times)):
        if numb % 100 == 0:
            print(f"Main process: query {numb} / {len(workload.queries)}")
        n_arr = start_time + a
        q.arrival_time = n_arr
        time.sleep(n_arr - time.perf_counter())
        
        match q.time_cls:
            case 0:
                q.queue_arrival_time = time.perf_counter()
                workload.queue_dct[fast_keys[fast_idx]].put(q)
                fast_idx = 0 if (fast_idx + 1) == len(fast_keys) else (fast_idx + 1)
            case 1:
                q.queue_arrival_time = time.perf_counter()
                workload.queue_dct[medium_keys[med_idx]].put(q)
                med_idx = 0 if (med_idx + 1) == len(medium_keys) else (med_idx + 1)
            case 2:
                q.queue_arrival_time = time.perf_counter()
                workload.queue_dct['slow'].put(q)
    for k in workload.queue_dct.keys():
        workload.queue_dct[k].put(None)
    with open(f"{path}/main.json", 'w') as f:
        f.write("done")
    exit()


def main_balance_runner_v2(sample_name, scale, url = 'http://172.21.233.23:8891/sparql', bl_type='planRGCN'):
    np.random.seed(42)
    random.seed(42)
    
    # Workload Setup
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df)
    print(df['mean_latency'].quantile(q=0.25))
    w = WorkloadV2()
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv")
    a = ArrivalRateDecider()
    w.shuffle_queries()
    w.shuffle_queries()
    #w.reorder_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu=44))
    
    start_time = time.perf_counter()
    print(start_time)
    f_lb =f'/data/{sample_name}/load_balance'
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    main_node = None
    match bl_type:
        case "planRGCN":
            main_node=main_worker
    
    procs = {
        "monitor" : multiprocessing.Process(target=monitor_queue, args=(w,start_time,path)),
        "main" : multiprocessing.Process(target=main_node, args=(w,start_time,path)),
        "slow" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"slow",url, start_time,path,)),
        "med1" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"med1",url, start_time,path,)),
        "med2" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"med2",url, start_time,path,)),
        "med3" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"med3",url, start_time,path,)),
        "fast1" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"fast1",url, start_time,path,)),
        "fast2" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"fast2",url, start_time,path,)),
        "fast3" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"fast3",url, start_time,path,)),
        "fast4" : multiprocessing.Process(target=execute_query_worker_v2, args=(w,"fast4",url, start_time,path,)),
    }
    
    for k in procs.keys():
        procs[k].start()
    
    for k in procs.keys():
        procs[k].join()
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time-start_time}")
    
def get_all_queries(b):
    queries = []
    for w in b.slow_queue.worker:
        queries.extend(w.past_queries)
    for w in b.med_queue.worker:
        queries.extend(w.past_queries)
    for w in b.fast_queue.worker:
        queries.extend(w.past_queries)
    return queries

def get_all_queriesFIFO(b):
    queries = []
    for w in b.fast_queue.worker:
        queries.extend(w.past_queries)
    return queries

def workload_latency(qs):
    total_latency = 0
    for q in qs:
        total_latency += (q.finish_time- q.arrivaltime)
    return total_latency
def relative_improvement(wl, bwl):
    return (bwl-wl)/bwl


def execute_query_worker_v2(workload:WorkloadV2, w_type, url, start_time, path):
    data = []
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    #sparql.setTimeout(1800)
    sparql.setTimeout(900)
    while True:
        val = workload.queue_dct[w_type].get()
        if val is None:
            break
        q = val
        
        q_start_time = time.perf_counter()
        #execute stuff
        ret = execute_query(q.query_string,sparql, timeout=900)
        #debug code /simulation
        """if w_type.startswith('fast'):
            time.sleep(0.001)
        elif w_type.startswith('med'):
            time.sleep(1)
        elif w_type.startswith('slow'):
            time.sleep(10)
        ret = None"""
        q_end_time = time.perf_counter()
        elapsed_time = q_end_time-q_start_time
        data.append({
            'query': str(q), 
            'start_time':start_time, 
            'arrival_time': q.arrival_time, 
            'queue_arrival_time':q.queue_arrival_time, 
            'query_execution_start': q_start_time, 
            'query_execution_end': q_end_time, 
            'execution_time': elapsed_time, 
            'response': 'ok' if ret is not None else 'not ok'})
        if len(data) % 100 == 0:
            with open(f"{path}/{w_type}_{str(len(data))}.json", 'w') as f:
                json.dump(data,f)
            
    with open(f"{path}/{w_type}.json", 'w') as f:
        json.dump(data,f)
    exit()

def execute_query_worker_FIFO(workload:Workload, w_type, url, start_time, path):
    w_str = w_type
    data = []
    #debug varible
    #qs = []
    while True:
        val = workload.FIFO_queue.get()
        if val is None:
            break
        (q,ar) = val
        n_ar = ar+start_time
        while( time.perf_counter()< n_ar):
            pass
        q_start_time = time.perf_counter()
        #execute stuff
        ret = execute_query(q.query_string,url)
        q_end_time = time.perf_counter()
        elapsed_time = q_end_time-q_start_time
        print(w_type+ '  ', len(data))
        #qs.append(q)
        data.append({'query': str(q), 'start_time':start_time, 'arrival_time': n_ar, 'query_start_time': q_start_time, 'query_end_time': q_end_time, 'elapsed_time': elapsed_time, 'response': 'ok' if ret is not None else 'not ok'})
    with open(f"{path}/{w_str}.json", 'w') as f:
        json.dump(data,f)
    #debug variable
    """chosen_i = None
    for i, d in enumerate(qs):
        if d.true_time_cls == 2:
            chosen_i = i
            break
    print(w_type,chosen_i)"""
    exit()

def main_balance_runnerFIFO(sample_name, scale, url = 'http://172.21.233.23:8891/sparql'):
    np.random.seed(42)
    random.seed(42)
    
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
    url = "http://172.21.233.14:8891/sparql"
    
    # Workload Setup
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df)
    print(df['mean_latency'].quantile(q=0.25))
    w = Workload()
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv")
    a = ArrivalRateDecider()
    w.shuffle_queries()
    w.shuffle_queries()
    w.reorder_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu=1000))
    w.initialise_FIFO_que()
    for _ in range(8):
        w.FIFO_queue.put(None)
    start_time = time.perf_counter()
    print(start_time)
    f_lb =f'/data/{sample_name}/load_balance_FIFO'
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = [
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"slow",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med1",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med2",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med3",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast1",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast2",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast3",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast4",url, start_time,path,)),
    ]
    
    for k in procs:
        k.start()
    
    for k in procs:
        k.join()
    end_time = time.perf_counter()
    print(f"elapsed time: {end_time-start_time}")