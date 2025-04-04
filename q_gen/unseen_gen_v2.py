import sys

from q_gen.unseen_gen import SimpleQuery

sys.path.append('/PlanRGCN/')
import os

os.environ['QG_JAR'] = '/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR'] = '/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'
import time

from q_gen.pp_gen import PPGenerator
from q_gen.util import Utility

import json

import pickle

import pandas as pd


from feature_extraction.sparql import Endpoint
from graph_construction.jar_utils import get_ent_rel


class UnseenGenerator:
    """
    Generated unseen queries for a RDF store at a specific endpoint
    """

    def __init__(self, train_file, val_file, test_file, pred_stat_path, url, outputfolder, subj_stat_path, new_qs_folder):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.endpoint = Endpoint(url)
        self.endpoint.sparql.setTimeout(900)
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.val_df = pd.read_csv(self.val_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file, sep='\t')


        self.template1_output = os.path.join(outputfolder, "filter_qs")
        os.makedirs(self.template1_output, exist_ok=True)

        print('Extracting entities and relations from training set')
        train_val_rels, train_val_ents = self.get_ent_rels_from_train_val()
        print('Loading predicate frequencies')
        self.pred_freq = self.get_pred_freq(pred_stat_path)
        print(('Filtering for unseen predicarte'))

        sorted_rels, self.pred_freq_unseen = self.get_pred_unseen_freq(outputfolder, train_val_rels)

        print(('Extract Valuable Rels'))
        valuable_rels = self.get_rels_usable_in_values(train_val_rels, outputfolder)
        # print(('Extract Valuable Rels (unseen)'))
        # unseen_valuable_rels = self.get_rels_usable_in_values(sorted_rels, outputfolder, output='unseen')

        self.single_tp_qs = os.path.join(outputfolder, "singleTPqs")
        above10sec = os.path.join(self.single_tp_qs, 'above10sec')
        #bet_1_10sec = os.path.join(self.single_tp_qs, 'bet1_10sec')
        #less1sec = os.path.join(self.single_tp_qs, 'less1sec')

        #bet_df = self.formated_generated_evaluated_qs(bet_1_10sec, 2, 8)
        #bet_output_path = os.path.join(self.single_tp_qs, '1_10_unseen_executions.tsv')
        # bet_df.to_csv(bet_output_path, sep='\t', index=False)

        # benchmarking slow queries
        # self.benchmark_slow_qs(above10sec, url)

        slow_df = self.formated_generated_evaluated_qs(above10sec, 15, 700, rt_int='slow')

        bet_df = self.manually_defined_qs('/PlanRGCN/data/dbpedia_queries/manualUnseenDBpedia1_10')

        # all_output_path = os.path.join(self.single_tp_qs, 'unseen_executions_and_test_set.tsv')
        all_df = pd.concat([bet_df, slow_df])
        self.output_new_pp_qs(all_df, new_qs_folder, test_file)
        print(f'Succesfully generated {len(all_df)} qs')

        #rel_str = self.get_rel_list_clause(valuable_rels)

    def formated_generated_evaluated_qs(self, bet_1_10sec, lower_thres, upper_thres, rt_int='med'):
        be_qs = []
        for q_idx in os.listdir(bet_1_10sec):
            q = SimpleQuery(os.path.join(bet_1_10sec, q_idx))
            be_qs.append({'queryString': q.query_text,
                          'mean_latency': q.latency,
                          'ResultsSetSize': q.cardinality,
                          'queryID': f"http://lsq.aksw.org/CompletelyUnseenSimpleTP_{rt_int}_{q_idx}"})
        bet_df = pd.DataFrame(be_qs)
        bet_df['id'] = bet_df['queryID']
        bet_df['resultset_0'] = bet_df['ResultsSetSize']
        bet_df = bet_df[(bet_df.mean_latency > lower_thres) & (bet_df.mean_latency < upper_thres)]
        bet_df = self.reorder_df_to_pred_format(bet_df)
        return bet_df


    def output_new_pp_qs(self, unseen_df, new_qs_folder, test_file):
        base_test_df = pd.read_csv(test_file, sep='\t')
        assert len(set(base_test_df.columns).intersection(list(unseen_df.columns)))
        new_df = pd.concat([base_test_df, unseen_df])
        assert (len(unseen_df)+len(base_test_df)) == len(new_df)
        assert not os.path.exists(new_qs_folder)
        os.makedirs(new_qs_folder)
        new_df.to_csv(os.path.join(new_qs_folder, 'queries.tsv'), sep='\t', index=False)




    def reorder_df_to_pred_format(self, df: pd.DataFrame):
        """
        Reorders the df to be on the same format as test_sampled.tsv such that inference functionality can be used
        @param df:
        @return: df
        """
        cols = ['id', 'queryString', 'query_string_0', 'latency_0', 'resultset_0',
                'query_string_1', 'latency_1', 'resultset_1', 'query_string_2',
                'latency_2', 'resultset_2', 'mean_latency', 'min_latency',
                'max_latency', 'time_outs', 'path', 'triple_count', 'subject_predicate',
                'predicate_object', 'subject_object', 'fully_concrete', 'join_count',
                'filter_count', 'left_join_count', 'union_count', 'order_count',
                'group_count', 'slice_count', 'zeroOrOne', 'ZeroOrMore', 'OneOrMore',
                'NotOneOf', 'Alternative', 'ComplexPath', 'MoreThanOnePredicate',
                'queryID', 'Queries with 1 TP', 'Queries with 2 TP',
                'Queries with more TP', 'S-P Concrete', 'P-O Concrete', 'S-O Concrete']
        for x in cols:
            if x not in df.columns:
                df[x] = None
        return df[cols]

    def get_pred_unseen_freq(self, outputfolder, train_val_rels):
        unseen_path = os.path.join(outputfolder, 'pred_freq_unseen.pickle')
        if not os.path.exists(unseen_path):
            pred_freq_unseen = {}
            for x in [x for x in self.pred_freq.keys() if x not in train_val_rels]:
                pred_freq_unseen[x] = self.pred_freq[x]
            sorted_rels = sorted(list(pred_freq_unseen.keys()), reverse=True,
                                 key=lambda x: pred_freq_unseen[x])
            with open(unseen_path, 'wb') as f:
                pickle.dump((pred_freq_unseen, sorted_rels), f)
        else:
            with open(unseen_path, 'rb') as f:
                pred_freq_unseen, sorted_rels = pickle.load(f)
        return sorted_rels, pred_freq_unseen

    def get_rel_list_clause(self, valuable_rels):
        rel_str = '( '
        for i, x in enumerate(valuable_rels):
            if i < (len(valuable_rels) - 1):
                rel_str += f'<{x}>, '
            else:
                rel_str += f'<{x}> '
        rel_str += ' )'
        return rel_str

    def get_rels_usable_in_values(self, rels, out_t_rel_folder, output='train'):
        pickle_path = os.path.join(out_t_rel_folder, f'{output}_rels_in_values_clause.pickle')
        if not os.path.exists(pickle_path):
            def get_train_pred_str_single_rel(rel):
                t_ent_str = "{"
                t_ent_str += f" <{rel}> "
                t_ent_str += "}"
                return t_ent_str

            # check which rels are hard to process in values clause
            valuable_rels = set()
            for i, r in enumerate(rels):
                query = f"""
                SELECT ?pred2 WHERE {{
                    ?s ?pred2 ?o2 .  

                    VALUES ?pred2 {get_train_pred_str_single_rel(r)}   
                }}
                LIMIT 1
                """
                try:
                    self.endpoint.run_query(query)
                    valuable_rels.add(r)
                except Exception as e:
                    print(e)
            with open(pickle_path, 'wb') as f:
                pickle.dump(valuable_rels, f)
        else:
            with open(pickle_path, 'rb') as f:
                valuable_rels = pickle.load(f)
        return valuable_rels

    def get_ent_rels_from_train_val(self):
        return Utility.get_ent_rels_from_train_val(self.train_df, self.val_df)

    def get_pred_freq(self, pred_stat_path):
        pred_freq_path = os.path.join(self.template1_output, 'pred_freq.pickle')
        if not os.path.exists(pred_freq_path):
            pred_freq = Utility.get_pred_freq(pred_stat_path)
            with open(pred_freq_path, 'wb') as f:
                pickle.dump(pred_freq, f)
            return pred_freq

        with open(pred_freq_path, 'rb') as f:
            pred_freq = pickle.load(f)
            return pred_freq

    def get_subj_freq(self, subj_stat_path):
        return Utility.get_subj_freq(subj_stat_path)

    def star_template1(self, pred1, pred2, pred3):
        return f"""
        SELECT ?s ?o ?o2 ?o3 WHERE 
        {{  ?s {pred1} ?o .
            ?s {pred2} ?o2 .
            ?s {pred3} ?o3 .
        }} 
        """

    def manually_defined_qs(self, query_folder):
        queries = []
        for i, p in enumerate(os.listdir(query_folder)):
            fp = os.path.join(query_folder, p)
            q ={}
            with open(fp, 'r') as f:
                q['queryString'] = f.read()
                q['queryID'] = f'http://lsq.aksw.org/ManuallyGeneratedUnseen{i}'
                q['id'] = q['queryID']
                start = time.time()
                res = self.endpoint.run_query_and_results(q['queryString'])
                dur = time.time()-start
                card = len(res)
                q['mean_latency'] = dur
                q['resultset_size'] = card
                queries.append(q)

        df = pd.DataFrame(queries)
        df['resultset_1'] = df['resultset_size']
        df = self.reorder_df_to_pred_format(df)
        return df


if __name__ == "__main__":
    base_path = '/data/DBpedia_3_class_full'
    train_file = f'{base_path}/train_sampled.tsv'
    val_file = f'{base_path}/val_sampled.tsv'
    test_file = f'{base_path}/test_sampled.tsv'
    url = 'http://172.21.233.14:8891/sparql'
    url = 'http://172.21.233.23:8891/sparql'
    pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
    subj_stat_path = '/data/metaKGStat/dbpedia/entity/ent_stat/batches_response_stats/subj'
    outputfolder = '/data/generatedUnseen'
    new_qs_folder = '/data/DBpedia_3_class_full/newUnseenQs3'
    os.makedirs(outputfolder, exist_ok=True)
    UnseenGenerator(train_file, val_file, test_file, pred_stat_path, url, outputfolder, subj_stat_path,new_qs_folder)