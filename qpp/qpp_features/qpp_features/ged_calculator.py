import argparse
import json
import time
import pandas as pd
import os, math
import multiprocessing
from qpp_features.database import DatabaseConnector

def get_id_combinations(ids, k=5000, database_path="/qpp/ged_dbpedia.db", check_db=True):
    """ This is a generator now.
    """
    if check_db:
        db = DatabaseConnector(file_name=database_path)
    s = set()
    for i in range(len(ids)):
        for j in range(i, len(ids)):
            if check_db:
                if db.exists(ids[i], ids[j]):
                    continue  
            s.add(IDPair(ids[i], ids[j]))
            if check_db:
                db.insert_ids(ids[i], ids[j])
            
            if len(s) == k:
                yield s
                s = set()
    yield s


def create_id_combination_files(ids, keep=False, k=10000):
    keep = False
    counter = 1
    input_files = []
    output_files = []
    if keep is True:
       s = get_id_combinations(ids, k=k)
       d = list(next(s))
    else:
        d = []
    while len(d) > 0 and keep:
        #raise Exception("Should not reach this code")
        if len(d) == 0:
            keep = False
            break
        fp = os.path.join(comb_path, f"comb_{counter}.json")
        dist_fp = os.path.join(dist_path, f"comb_{counter}.json")
        input_files.append(fp)
        output_files.append(dist_fp)
        json.dump(create_json(d, query_df), open(fp, "w"))
        counter += 1
        print(f"Finished processing {fp} with {len(d)}")
        #if len(d) < k:
        #    break
        try:
           d = list(next(s))
        except Exception:
            break

def get_id_combinations_mult(ids, k=5000, database_path="/qpp/ged_dbpedia.db", check_db=False, index_start=0, index_end=  -1):
    """ This is a generator now.
    """
    if check_db:
        db = DatabaseConnector(file_name=database_path)
    s = set()
    
    if index_end == -1:
        raise Exception("You need to specify interval range")
    
    for i in range(index_start, index_end):
        for j in range(i+1, len(ids)):
            if check_db:
                if db.exists(ids[i], ids[j]):
                    continue  
            s.add(IDPair(ids[i], ids[j]))
            if check_db:
                db.insert_ids(ids[i], ids[j])
            
            if len(s) == k:
                yield s
                s = set()
    yield s

def calculate_intervals(n, sample_size):
    intervals = []
    prev = 0
    intervals.append((prev,prev+sample_size))
    prev = sample_size
    while (prev <= n):
        increment = prev+sample_size
        if increment< n:
            intervals.append((prev, increment))
        else:
            intervals.append((prev, n))
        prev += sample_size
    return intervals
    
def process_single_combination(ids, k, database_path, check_db, index_start, index_end, comb_path, query_df):
        s = get_id_combinations_mult(ids, k=k, database_path=database_path, check_db=check_db, index_start=index_start, index_end =index_end)

        counter=0
        d = list(next(s))
        while len(d) > 0:
            fp = os.path.join(comb_path, f"comb_{str(index_start)}_{str(index_end)}_{counter}.json")
            #dist_fp = os.path.join(dist_path, f"comb_{counter}.json")
            json.dump(create_json(d, query_df), open(fp, "w"))
            counter += 1
            print(f"Finished processing {fp} with {len(d)}")
            try:
               d = list(next(s))
            except Exception:
                #This is should after reaching the last iteration.
                break
def create_id_combination_files_multi_processing(ids, query_df,comb_path,keep=True, k=10000, n_cpus=1):
    
    n_ids = len(ids)
    sample_size = math.ceil(n_ids/n_cpus) #we want integer here.
    intervals = calculate_intervals(n_ids, sample_size)
    index_starts, index_ends = [],[]
    for i in intervals:
        index_starts.append(i[0])
        index_ends.append(i[1])
    start = time.time()
    starts = [start for x in range(len(intervals))]
    idss = [ids for x in range(len(intervals))]
    ks = [k for x in range(len(intervals))]
    dp = [None for x in range(len(intervals))]
    check_dbs = [False for x in range(len(intervals))]
    comb_paths = [comb_path for x in range(len(intervals))]
    query_dfs = [query_df for x in range(len(intervals))]
    lst = list(
        zip(idss,ks, dp,check_dbs, index_starts, index_ends,comb_paths, query_dfs)
    )
    #pool = multiprocessing.Pool(cpus)
    with multiprocessing.Pool(n_cpus) as pool:
        M = pool.starmap(process_single_combination, lst)
        pool.close()
        pool.join()
        
            
def m_proc_pair_calc(ids,keep=True, k=10000,cpus=22):
    n_ids = len(ids)
    intervals = calculate_intervals(n_ids, sample_size)
    sample_size = math.ceil(n_ids/n_cpus) #we want integer here.
    
    # lst = list(zip(input_files, output_files, [None for x in range(len(output_files))]))
    start = time.time()
    lst = list(
        zip(ids, output_files, [start for x in range(len(output_files))])
    )
    #pool = multiprocessing.Pool(cpus)
    with multiprocessing.Pool(cpus) as pool:
        M = pool.starmap(calculate_dists, lst)
        pool.close()
        pool.join()


def mult_calculate_dists(comb_path,dist_path, cpus):
    input_files = [os.path.join(comb_path,x) for x in os.listdir(comb_path)]
    output_files = [os.path.join(dist_path,x) for x in os.listdir(comb_path)]
    multi_process_data(input_files, output_files, cpus=cpus)
    



def create_json(id_pairs, df):
    lst = []
    for pair in id_pairs:
        """dct = {
            "queryID1": pair.id1,
            "queryID2": pair.id2,
            "queryString1": df.loc[pair.id1, "queryString"],
            "queryString2": df.loc[pair.id2, "queryString"],
        }"""
        dct = {
            "queryID1": pair.id1,
            "queryID2": pair.id2,
            "queryString1": df.loc[pair.id1, "query_string_0"],
            "queryString2": df.loc[pair.id2, "query_string_0"],

        }

        lst.append(dct)
    return lst


class IDPair:
    """Wrapper class for queryID string"""

    def __init__(self, id1, id2) -> None:
        self.id1 = id1
        self.id2 = id2

    def __eq__(self, __value: object) -> bool:
        if (self.id1 == __value.id1) and (self.id2 == __value.id2):
            return True
        if (self.id2 == __value.id1) and (self.id1 == __value.id2):
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.id1) + hash(self.id2)

    def __str__(self) -> str:
        return f"{self.id1},{self.id2}"



# files, output_files,
def multi_process_data(input_files, output_files, cpus=3):
    # lst = list(zip(input_files, output_files, [None for x in range(len(output_files))]))
    start = time.time()
    lst = list(
        zip(input_files, output_files, [start for x in range(len(output_files))])
    )
    #pool = multiprocessing.Pool(cpus)
    with multiprocessing.Pool(cpus) as pool:
        M = pool.starmap(calculate_dists, lst)
        pool.close()
        pool.join()


def calculate_dists(file, output_file, start):
    if start is None:
        start = time.time()
    jarpath = "/PlanRGCN/qpp/jars/sparql-query2vec-0.0.1.jar"

    os.system(
        f"java -jar {jarpath} ged-opt --input-queryfile={file} --output-queryfile={output_file}"
    )
    if os.path.exists(output_file):
       os.remove(file)
    print("Finishing in ", file, output_file, time.time() - start)


def work_func(x):
    """for testing purposes

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("work_func:", x, "PID", os.getpid())
    time.sleep(2)
    return x**5

def main(query_log_file, output_folder,task, k=100000, cpus=20):
    query_df = pd.read_csv(query_log_file, sep="\t")
    #ids = list(query_df["queryID"].unique())
    ids = list(query_df["id"].unique())
    print("Amount of unique IDs ", len(ids))
    #query_df = query_df.set_index("queryID")
    query_df = query_df.set_index("id")

    comb_path = os.path.join(output_folder, "combinations")
    os.makedirs(comb_path, exist_ok=True)

    dist_path = os.path.join(output_folder, "distances")
    os.makedirs(dist_path, exist_ok=True)

    
    #create_id_combination_files(ids)
    #m_proc_pair_calc(ids, keep=True, k=k, cpus=cpus)
    match task:
        case "combinations":
            create_id_combination_files_multi_processing(ids, query_df,comb_path,keep=True, k=k, n_cpus=cpus)
        case "dist_calc":
            mult_calculate_dists(comb_path,dist_path, cpus)
    #os.system(f"rm -rf {comb_path}")

if __name__ == "__main__":
    #python3 -m qpp_features.ged_calculator /data/DBpedia2016h_weight_loss/all.tsv /data/dbpedia_dist2 
    #python3 -m qpp_features.ged_calculator /data/DBpedia2016h_weight_loss/all.tsv /data/dbpedia_dist2 -t dist_calc
    # Create an ArgumentParser
    parser = argparse.ArgumentParser(description="Calculate and save distance matrix.")

    # Add input and output file arguments
    parser.add_argument("query_log_file", type=str, help="Input query log file")
    parser.add_argument("output_folder", type=str, help="Output distance matrix folder")
    parser.add_argument("-t", type=str, help="task", default='combinations', choices=["combinations", "dist_calc"])


    # Parse the command line arguments
    args = parser.parse_args()
    query_log_file = args.query_log_file
    output_folder = args.output_folder
    task = args.t
    
    main(query_log_file, output_folder, task, cpus=20)
