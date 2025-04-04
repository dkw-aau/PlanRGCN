import os
import time

from q_gen.util import Utility

os.environ['QG_JAR']='/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
import json

import pickle

import pandas as pd

from feature_extraction.sparql import Endpoint
from graph_construction.jar_utils import get_ent_rel




class PPGenerator:
    def __init__(self, train_file, val_file, test_file, pred_stat_path,url, outputfolder, subj_stat_path):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.endpoint = Endpoint(url)
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.val_df = pd.read_csv(self.val_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file, sep='\t')


        train_val_rels, train_val_ents = self.get_ent_rels_from_train_val()
        self.pred_freq =  self.get_pred_freq(pred_stat_path)

        #self.extract_PP_cand(train_val_rels, outputfolder, descending=True) #currently running on server 20

        #self.extract_PP_cand(train_val_rels, outputfolder)
        #pp_cands = self.load_pp_cand_from_file(outputfolder,file='/data/generatedPP/PP_candidates.pickle')

        tv_ents, tv_rels = self.get_ent_rels_from_train_val()
        pp_cands = self.load_pp_cand_from_file(outputfolder)

        subj_freq = self.get_subj_freq(subj_stat_path)

        self.generate_opt_PP_qs(outputfolder, pp_cands, url)

    def generate_opt_PP_qs(self, outputfolder, pp_cands, url):
        queryFolder = os.path.join(outputfolder, 'PP_w_Optionals')
        above1sec = os.path.join(queryFolder, 'above1sec')
        under1sec = os.path.join(queryFolder, 'under1sec')
        os.makedirs(queryFolder, exist_ok=True)
        os.makedirs(above1sec, exist_ok=True)
        os.makedirs(under1sec, exist_ok=True)
        self.endpoint2 = Endpoint(url)
        self.endpoint2.sparql.setTimeout(1)
        f_queries = 0
        m_s_queries = 0
        q_gen_no = 1

        for pp_cand_test in pp_cands:
            print(f'Beginning on {pp_cand_test}')
            start_cands = self.endpoint.run_query(f"""
            SELECT ?S ?pred WHERE {{ 
            ?S <{pp_cand_test}> ?O .
            ?O <{pp_cand_test}> ?O2 .
            ?O ?pred ?e343 .
                {{
                    SELECT COUNT(*) AS ?count {{
                        ?S ?p ?e
                    }}
                }}
                {{
                    SELECT COUNT(*) AS ?count2 {{
                        ?O ?p1421 ?e124354
                    }}
                }}
                FILTER( ?count2 > 20 )
                FILTER( ?count > 20 )
            }} 
            LIMIT 20""")

            subj_cands = [x['S']['value'] for x in start_cands['results']['bindings']]
            trp_pred_cands = list(set([x['pred']['value'] for x in start_cands['results']['bindings']]))
            optionals = ""
            for idx, opt_cand in enumerate(trp_pred_cands[:6]):
                optionals += f"""
                OPTIONAL {{
                ?e <{opt_cand}> ?e11{idx} .
                }}
                """

            #time.sleep(60)
            subj_can_test = subj_cands[1]
            for subj_can_test in subj_cands:
                q_subj = f"""
                SELECT ?e ?e2 WHERE {{
                <{subj_can_test}> <{pp_cand_test}>+ ?e .
                {optionals}
                }}
                """
                try:
                    res, dur = self.endpoint2.time_and_run_query(q_subj)
                    q_folder = os.path.join(under1sec, str(q_gen_no))
                    os.makedirs(q_folder, exist_ok=True)
                    with open(os.path.join(q_folder, 'query.txt'), 'w') as f:
                        f.write(q_subj)
                    with open(os.path.join(q_folder, 'duration.txt'), 'w') as f:
                        f.write(str(dur) + "  s")
                    f_queries += 1
                    q_gen_no += 1
                except TimeoutError:
                    q_folder = os.path.join(above1sec, str(q_gen_no))
                    os.makedirs(q_folder, exist_ok=True)
                    with open(os.path.join(q_folder, 'query.txt'), 'w') as f:
                        f.write(q_subj)
                    q_gen_no += 1
                    m_s_queries += 1
                print(f'Generated stats: med/slow qs: {m_s_queries}, fast queries: {f_queries}')

    def load_pp_cand_from_file(self, outputfolder, file=None):
        if file is None:
            pp_can_path = os.path.join(outputfolder, 'PP_candidates.pickle')
        else:
            pp_can_path = file
        with open(pp_can_path, 'rb') as f:
            pps = pickle.load(f)
            return pps

    def get_ent_rels_from_train_val(self):
        return Utility.get_ent_rels_from_train_val(self.train_df, self.val_df)

    def get_pred_freq(self, pred_stat_path):
        return Utility.get_pred_freq(pred_stat_path)

    def get_subj_freq(self, subj_stat_path):
        return Utility.get_subj_freq(subj_stat_path)

    def extract_PP_cand(self, train_val_rels, outputfolder, descending=False):
        sorted_rels = sorted(list(self.pred_freq.keys()), key=lambda x: self.pred_freq[x], reverse=descending)
        sorted_rels = [x for x in sorted_rels if x not in train_val_rels]
        sorted_rels_50k = sorted_rels[:100000]
        pp_candidates = []
        for idx, rel in enumerate(sorted_rels_50k):
            if idx % 50 == 0:
                print(f"processd {idx} of {len(sorted_rels_50k)}")
            if self.is_rel_PP_legilable(rel):
                pp_candidates.append(rel)
            if len(pp_candidates) % 20 == 0:
                pp_can_path = os.path.join(outputfolder, 'PP_candidates.pickle')
                with open(pp_can_path, 'wb') as f:
                    pickle.dump(pp_candidates, f)
        if len(pp_candidates) > 0:
            pp_can_path = os.path.join(outputfolder, 'PP_candidates.pickle')
            with open(pp_can_path, 'wb') as f:
                pickle.dump(pp_candidates, f)



    def PP_template1(self, pp_pred, pred, ent):
        return f"""
        SELECT ?e ?o2 WHERE {{
        <ent> <{pp_pred}>+ ?e .
        ?e <{pred}> ?o2 
        }}
        """

    #Use when additional TP has literal values.
    def PP_template1_w_filter(self, pp_pred, pred, ent):
        return f"""
        SELECT ?e ?o2 WHERE {{
        <ent> <{pp_pred}>+ ?e .
        ?e <{pred}> ?o2 
        FILTER( LANG(?o) == \"\"en\"\")
        }}
        """

    def PP_template1_w_optional(self, pp_pred, pred, ent):
        return f"""
        SELECT ?e ?o2 WHERE {{
        <ent> <{pp_pred}>+ ?e .
        OPTIONAL {{
        ?e <{pred}> ?o2
        }} 
        FILTER( LANG(?o) == \"\"en\"\")
        }}
        """

    def is_rel_PP_legilable(self,rel):
        q = f"""
        ASK {{
        ?s <{rel}> ?e1 . ?e1 <{rel}> ?e2 .
        }}
        """
        return self.endpoint.run_query(q)['boolean']

class SimplePPGen:
    def __init__(self, pred, endpoint:Endpoint):
        self.pred = pred
        self.endpoint = endpoint

    def find_start_entity_cand(self):
        start_cands = self.endpoint.run_query(f"SELECT DISTINCT * WHERE {{ ?S <{self.pred}> ?O }} LIMIT 20")
        pass


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
    gen = PPGenerator(train_file, val_file, test_file,pred_stat_path, url, outputfolder, subj_stat_path)








def codeCrashingDB():
    pp_cand_test = '<somepRed>'
    start_cands = f"""
                    SELECT ?S ?O ?p1 WHERE {{ 
                    ?S <{pp_cand_test}> ?O .
                    ?O ?p1 ?e2 .
                    }} 
                    GROUP BY ?S ?O ?p1
                    HAVING (COUNT(?e2) > 20)
                    LIMIT 20"""