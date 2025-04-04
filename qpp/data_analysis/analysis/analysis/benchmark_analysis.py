import matplotlib.pyplot as plt
import pandas as pd
import json, importlib
import numpy as np, os
import analysis.benchmark_analysis as b

def load_run(fp:str):
    data = json.load(open(fp,'r'))
    assert len(data['ids']) == len(data['queries']) and len(data['queries']) == len(data['latency'])
    #run_df = pd.DataFrame(data)
    return data

def reorder_lats(order, run):
    o = {}
    for i,x in enumerate(run['ids']):
        o[x] = i
    lats = []
    for x in order:
        lats.append(run['latency'][o[x]])
        assert run['ids'][o[x]] == x
    return lats    

def load_bench(path='/qpp/dataset/sampled_data_20000/bench/', iterations=10):
    run = load_run(f'{path}run_0.json')
    run['latency_0'] = run['latency']
    del run['latency']
    order = run['ids']
    for i in range(1,iterations):
        run_i = load_run(f'{path}run_{i}.json')
        run[f'latency_{i}'] = reorder_lats(order, run_i)
    return pd.DataFrame(run)

def calculate_mean_stats(run_df :pd.DataFrame):
    lat_cols = [x for x in run_df.columns if 'latency' in x]
    
    mean_lat = []
    min_lats = []
    max_lats = []
    time_outs = []
    for index,row in run_df.iterrows():
        lat_arr = []
        t_o = 0
        for col in lat_cols:
            if row[col]!= 'timeout':
                lat_arr.append(row[col])
            else:
                t_o += 1
        time_outs.append(t_o)
        mean_lat.append(np.mean(lat_arr))
        min_lats.append(np.min(lat_arr))
        max_lats.append(np.max(lat_arr))
    run_df['mean_latency'] = mean_lat
    run_df['min_latency'] = min_lats
    run_df['max_latency'] = max_lats
    run_df['time_outs'] = time_outs
    return run_df
def merge_bench_query_log(bench_df, query_log_df: pd.DataFrame):
     bench_df = bench_df.set_index('ids')
     query_log_df = query_log_df[query_log_df['queryID'].isin(bench_df.index)]
     mean_lat = []
     min_lats = []
     max_lats = []
     for i, row in query_log_df.iterrows():
        mean_lat.append(  bench_df.loc[row['queryID']]['mean_latency'] ) 
        min_lats.append(  bench_df.loc[row['queryID']]['min_latency'] ) 
        max_lats.append(  bench_df.loc[row['queryID']]['max_latency'] ) 
     print(type(mean_lat[0]))
     query_log_df['mean_latency'] = mean_lat
     query_log_df['min_latency'] = min_lats
     query_log_df['max_latency'] = max_lats
     return query_log_df
 
def extracted_interval_check(df:pd.DataFrame, column='mean_latency', file=None):
    if file is None:
        print("Extracted Interval Check")
        print("\t [0     :   0.01]      "+str(len(df[df[column]< 0.01])) + f"[{round(len(df[df[column]< 0.01])/len(df),2)}%]" )
        print("\t [0.01  :    0.1]      "+str(len(df[(0.01<df[column])&(df[column]<0.1)])) + f"[{round(len(df[(0.01<df[column])&(df[column]<0.1)])/len(df),2)}%]" )
        print("\t [0.1   :      1]      "+str(len(df[(0.1<df[column])&(df[column]<1)]))    + f"[{round(len(df[(0.1<df[column])&(df[column]<1)])/len(df),2)}%]"    )
        print("\t [1     :     10]      "+str(len(df[(1<df[column])&(df[column]<10)]))     + f"[{round(len(df[(1<df[column])&(df[column]<10)])/len(df),2)}%]"     )
        print("\t [10    :    100]      "+str(len(df[(10<df[column])&(df[column]<100)]))   + f"[{round(len(df[(10<df[column])&(df[column]<100)])/len(df),2)}%]"   )
        print("\t [100   :    inf]      "+str(len(df[df[column]> 100]))                    + f"[{round(len(df[df[column]> 100])/len(df),2)}%]"                    )
    else:
        f = open(file,'w')
        f.write("Interval\tQueries\tpercentage of Query Log")
        f.write('\n')
        f.write("[0 : 0.01]\t"+str(len(df[df[column]< 0.01])) + f"\t{round(len(df[df[column]< 0.01])/len(df),2)}%" )
        f.write('\n')
        f.write("[0.01 : 0.1]\t"+str(len(df[(0.01<df[column])&(df[column]<0.1)])) + f"\t{round(len(df[(0.01<df[column])&(df[column]<0.1)])/len(df),2)}%" )
        f.write('\n')
        f.write("[0.1 : 1]\t"+str(len(df[(0.1<df[column])&(df[column]<1)]))    + f"\t{round(len(df[(0.1<df[column])&(df[column]<1)])/len(df),2)}%"    )
        f.write('\n')
        f.write("[1 : 10]\t"+str(len(df[(1<df[column])&(df[column]<10)]))     + f"\t{round(len(df[(1<df[column])&(df[column]<10)])/len(df),2)}%"     )
        f.write('\n')
        f.write("[10 : 100]\t"+str(len(df[(10<df[column])&(df[column]<100)]))   + f"\t{round(len(df[(10<df[column])&(df[column]<100)])/len(df),2)}%"   )
        f.write('\n')
        f.write("[100 : inf]\t"+str(len(df[df[column]> 100]))                    + f"\t{round(len(df[df[column]> 100])/len(df),2)}%"                    )
        f.write('\n')
        f.close()
      
#dummy code for reloading in repl.
def reload():
  importlib.reload(json)
# We want statistics on
# - number of time out on queries
# - Boxplot of run times
# - Boxplot of average percentage deviation of runtimes from mean for queries.
#       - This is important to illustrate any issues or outliers of the benchmark.
# - Complete time to run a run of the benchmark. (Would be useful for future endeavors, reruns)

def make_merge(output_path='/qpp/dataset/sampled_data_20000/benchmark.tsv', bench_path='/qpp/dataset/sampled_data_20000/bench/'):
    query_log = pd.read_csv('/qpp/dataset/queries2015_2016_clean_w_stat_q_str.tsv', sep='\t')
    bench = b.load_bench(path=bench_path)
    bench = b.calculate_mean_stats(bench)
    query_log = b.merge_bench_query_log(bench, query_log)
    query_log.to_csv(output_path, sep='\t', index=False)

def return_merge(bench_path ='/bench/'):
    query_log = pd.read_csv('/qpp/dataset/queries2015_2016_clean_w_stat_q_str.tsv', sep='\t')
    bench = b.load_bench(path=bench_path)
    bench = b.calculate_mean_stats(bench)
    query_log = b.merge_bench_query_log(bench, query_log)
    return query_log

def print_benchmark_stats():
    print("20000 query sample dataset")
    bench = b.load_bench()
    bench = b.calculate_mean_stats(bench)
    b.extracted_interval_check(bench)
    
    print("Entire dataset")
    bench = b.load_bench(path='/bench/')
    bench = b.calculate_mean_stats(bench)
    b.extracted_interval_check(bench)

def find_timeout_queries(bench: pd.DataFrame, thres = 1800):
    qs = {}
    for i, row in bench.iterrows():
        for col in bench.columns:
            if 'lat' in col:
                if row[col] >= thres:
                    if row['ids'] in qs.keys():
                        qs[row['ids']].append(col)
                    else:
                        qs[row['ids']] = [col]
    return qs
def deviations(df:pd.DataFrame):
    cols =['latency_0', 'latency_1', 'latency_2', 'latency_3', 'latency_4', 'latency_5', 'latency_6', 'latency_7', 'latency_8', 'latency_9']
    for _, row in df.iterrows():
        lats = []
        for c in cols:
            lats.append(row[c])

def faster_running_qs(bench, lsq=1, mean_lat= 0.1):
    #query_log = b.merge_bench_query_log(bench, query_log)
    test_df = bench.loc[(bench['duration']>lsq) & (bench['mean_latency']<mean_lat) & (bench['resultCount']>=1)]
    for _, row in test_df.iterrows():
        yield row['queryString'], row['duration'], row['resultCount'], row['queryID']
    
    
def check_not_col(qs, col='latency_0'):
    for x in qs.keys():
     if not col in qs[x]:
      print(x, qs[x])
import analysis.sparql_util as s
def check_resultset_sizes(log_file):
    sparql = s.create_sparql()
    #query_log = pd.read_csv('/qpp/dataset/queries2015_2016_clean_w_stat_q_str.tsv', sep='\t')
    query_log = b.return_merge()
    gen = b.faster_running_qs(query_log)
    f = open(log_file, 'a')
    f.write('queryID,resultSet\n')
    for q,d,r,i in gen:
        ret = s.query(q,sparql)
        f.write(f'{i},{ret}\n')
        print(f'{i},{ret}\n')
    f.flush()
    f.close()

def make_runs_plot(bench, output='/qpp/dataset/DBpedia_2016_sampled/plots/'):
    col = []
    for c in bench.columns:
        if c.startswith('latency'):
            col.append(c)
    x = [i for i in range(len(bench))]
    plt.clf()
    for c in col:
        plt.plot(x,bench[c], label=c)
    plt.legend()
    plt.ylabel('Runtime (s)')
    plt.xlabel('Query')
    plt.title('Runs')
    plt.savefig(f"{output}run_plt.png")
    f = open(f'{output}runs_table.txt','w')
    f.write(str(bench[col].describe()))
    f.close()
    b.extracted_interval_check(bench, column='mean_latency', file=f'{output}mean_lat_interval.txt')
def plt_resultset():
    df = pd.read_csv('')


def get_chronological_bench(dir_path):
    files = sorted([x for x in os.listdir(dir_path) if x.endswith('.json')])
    all_lats = []
    for x in files:
        data = load_run(f"{dir_path}/{x}")
        all_lats.extend(data['latency'])
    return all_lats
def plot_bench(dir_path:str='/bench', output_path:str='/bench/plots'):
    plt.clf()
    all_lats = get_chronological_bench(dir_path)
    #print([i for i,x in enumerate(all_lats) if x > 0.003])
    plt.legend()
    plt.plot([x for x in range(len(all_lats))],all_lats, label='Latencies')
    plt.ylabel('Runtime (s)')
    plt.xlabel('Query Run')
    plt.title('Plot of all runs')
    plt.savefig(f"{output_path}/all_lat_plt.png")
    
if __name__ == "__main__":
    bench = b.load_bench()
    bench = b.calculate_mean_stats(bench)
    b.make_runs_plot(bench)
    #make_merge()
    """query_log = pd.read_csv('/qpp/dataset/queries2015_2016_clean_w_stat_q_str.tsv', sep='\t')
    bench = b.load_bench()
    bench = b.calculate_mean_stats(bench)
    query_log = b.merge_bench_query_log(bench, query_log)"""
    #print(query_log['resultCount'].describe())
    """print("20000 query sample dataset")
    bench = b.load_bench()
    b.extracted_interval_check(bench)
    
    print("Entire dataset")
    bench = b.load_bench(path='/bench/')
    bench = b.calculate_mean_stats(bench)
    b.extracted_interval_check(bench)"""
