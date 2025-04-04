import sys
sys.path.append('/PlanRGCN')
import os
import random
import time



os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR']='/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'
import os

import pandas as pd

from feature_extraction.sparql import Endpoint
from q_gen.util import Utility


class PPSel:
    def __init__(self, inputFolder, url=None, train_file = None, val_file = None, test_file = None, q_type_folder='PP_w_Optionals'):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.val_df = pd.read_csv(self.val_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file, sep='\t')

        self.q_type_folder=q_type_folder
        queryFolder = os.path.join(inputFolder, self.q_type_folder)
        self.above1sec = os.path.join(queryFolder, 'above1sec')
        self.under1sec = os.path.join(queryFolder, 'under1sec')
        self.above10sec = os.path.join(queryFolder, 'above10sec')
        self.between1_10sec = os.path.join(queryFolder, 'between1_10sec')
        self.other_error = os.path.join(queryFolder, 'other_error.txt')
        os.makedirs(self.above10sec, exist_ok=True)
        os.makedirs(self.between1_10sec, exist_ok=True)

        self.endpoint = Endpoint(url)
        self.endpoint.sparql.setTimeout(900)

        error_writer = open(self.other_error, 'a')

        tv_ents, tv_rels = Utility.get_ent_rels_from_train_val(self.train_df, self.test_df)

        random.seed(42)

        #Extract query directories and shuffle them such that the same relations are not too overused.
        query_dirs = list(os.listdir(self.above1sec))
        for i in range(3):
            random.shuffle(query_dirs)

        above_10_sec = []
        between1_10_sec = []
        tuples = []
        for number, query_dir in enumerate(query_dirs):
            query_path = os.path.join(self.above1sec, query_dir, 'query.txt')
            query_text = None
            with open(query_path, 'r') as f:
                query_text = f.read()
                query_text = query_text.replace('\n', ' ').replace('\t', '  ')
                queryID = f"http://lsq.aksw.org/{self.q_type_folder}_{self.above1sec}_{query_dir}"
                dur = 900
                try:
                    start = time.time()
                    res = self.endpoint.run_query(query_text)
                    res_size = len(res['results']['bindings'])
                    dur = time.time() - start
                except TimeoutError:
                    res_size = 0
                except Exception as e:
                    error_writer.write(f'Error for {queryID} with form: {query_text}, and error {repr(e)}\n\n')
                    continue
                entry = (queryID, query_text, dur, res_size)
                tuples.append(entry)
                if dur >= 10:
                    above_10_sec.append(entry)
                else:
                    between1_10_sec.append(entry)
            print(f"Currently processed queries: 1-10 -> {len(between1_10_sec)}, >10 -> {len(above_10_sec)}")
            if (number > 0) and (number % 10 == 0):
                #all queries in above 1 sec are executing somewhat
                pd.DataFrame(tuples, columns=['queryID', 'queryString', 'latency', 'resultsetSize']).to_csv( os.path.join(self.above1sec, 'query_executions.tsv'), sep='\t', index=False)
                # queries slower than 10 sec
                pd.DataFrame(above_10_sec, columns=['queryID', 'queryString', 'latency', 'resultsetSize']).to_csv(
                    os.path.join(self.above10sec, 'query_executions.tsv'), sep='\t', index=False)

                pd.DataFrame(between1_10_sec, columns=['queryID', 'queryString', 'latency', 'resultsetSize']).to_csv(
                    os.path.join(self.between1_10sec, 'query_executions.tsv'), sep='\t', index=False)
        error_writer.flush()
        error_writer.close()




if __name__ == "__main__":
    base_path = '/data/DBpedia_3_class_full'
    train_file = f'{base_path}/train_sampled.tsv'
    val_file = f'{base_path}/val_sampled.tsv'
    test_file = f'{base_path}/test_sampled.tsv'
    url= 'http://172.21.233.14:8891/sparql'
    pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
    subj_stat_path = '/data/metaKGStat/dbpedia/entity/ent_stat/batches_response_stats/subj'
    outputfolder = '/data/generatedPP'
    os.makedirs(outputfolder, exist_ok=True)

    query_selector = PPSel(outputfolder, url=url, train_file=train_file, val_file=val_file, test_file=test_file,q_type_folder='PP_w_Optionals')
