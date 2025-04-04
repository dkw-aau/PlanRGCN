import sys

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


def load_qs(path):
    """
    Path is generated between 1-10 sec optional queries from
    @param path:
    @return:
    """
    import pandas as pd
    df = pd.read_csv(path, sep='\t')
    # med queries
    df = df[(df['latency'] > 2) & (df['latency'] < 9)]
    df['qStrVirt'] = df.queryString.apply(lambda x: f"SPARQL {x} ;")
    print(df.qStrVirt.iloc[0])


class InstanceData1:
    def __init__(self, path):
        self.path = path
        inst_data_p = os.path.join(self.path, "inst_data.pickle")
        inst_query_p = os.path.join(self.path, "inst_query.rq")
        pred_p = os.path.join(self.path, "pred_file.txt")
        with open(inst_data_p, 'rb') as f:
            self.inst_data = pickle.load(f)

        with open(inst_query_p, 'r') as f:
            self.inst_query = f.read()

        with open(pred_p, 'r') as f:
            self.predicate = f.read()

    def inst_data_gen(self):
        for x in self.inst_data:
            pred1 = self.predicate
            pred2 = x['pred2']['value']
            pred3 = x['pred3']['value']
            yield pred1, pred2, pred3


class SimpleQuery:
    def __init__(self, path):
        self.path = path
        self.query_text = None
        self.latency = None
        self.cardinality = None

        card_dur_path = os.path.join(path, 'card_dur.txt')
        query_path = os.path.join(path, 'query_text.txt')
        with open(query_path) as f:
            self.query_text = f.read()

        with open(card_dur_path, 'r') as f:
            data = f.read()
            data = data.replace('(', '').replace(')', '')
            spl = data.split(',')
            self.latency = float(spl[1])
            self.cardinality = int(spl[0])


class UnseenGenerator:
    """
    Generated unseen queries for a RDF store at a specific endpoint
    """

    def __init__(self, train_file, val_file, test_file, pred_stat_path, url, outputfolder, subj_stat_path, new_qs_folder):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.endpoint = Endpoint(url)
        self.train_df = pd.read_csv(self.train_file, sep='\t')
        self.val_df = pd.read_csv(self.val_file, sep='\t')
        self.test_df = pd.read_csv(self.test_file, sep='\t')

        self.template1_output = os.path.join(outputfolder, "star_template_1")
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

        rel_str = self.get_rel_list_clause(valuable_rels)

        # This approach is too time consuming, but the data has been generated for 4 relations so let's look at them first.
        # self.instance_data_generation_1(rel_str, sorted_rels)
        # generates queries for inst data 1
        # self.query_gen_for_inst_data1()

        train_mid = self.train_df[(self.train_df['mean_latency'] > 1) & (self.train_df['mean_latency'] < 10.1)]
        mid_rel, mid_ents = Utility.get_ent_rels_from_train(train_mid)
        mid_rel_freqs = []
        for x in mid_rel:
            try:
                mid_rel_freqs.append(self.pred_freq[x])
            except:
                # If the relation in query does not exist in knowledge graph
                continue
        mid_rel_freqs = pd.Series(mid_rel_freqs)
        mid_rel_freqs.describe()
        # used to generate queries
        # self.generate_single_tp_queries(mid_rel_freqs, outputfolder, url)

        self.single_tp_qs = os.path.join(outputfolder, "singleTPqs")
        above10sec = os.path.join(self.single_tp_qs, 'above10sec')
        bet_1_10sec = os.path.join(self.single_tp_qs, 'bet1_10sec')
        less1sec = os.path.join(self.single_tp_qs, 'less1sec')

        bet_df = self.formated_generated_evaluated_qs(bet_1_10sec, 2, 8)
        bet_output_path = os.path.join(self.single_tp_qs, '1_10_unseen_executions.tsv')
        #bet_df.to_csv(bet_output_path, sep='\t', index=False)

        # benchmarking slow queries
        #self.benchmark_slow_qs(above10sec, url)

        slow_df = self.formated_generated_evaluated_qs(above10sec, 15, 700, rt_int='slow')

        #all_output_path = os.path.join(self.single_tp_qs, 'unseen_executions_and_test_set.tsv')
        all_df = pd.concat([bet_df, slow_df])
        self.output_new_pp_qs(all_df, new_qs_folder, test_file)
        print(f'Succesfully generated {len(all_df)} qs')

        """
        slow_qs = {}
        for no, q_idx in enumerate(os.listdir(above10sec)):
            query_path = os.path.join(above10sec, q_idx, 'query_text.txt')
            query_text = None
            with open(query_path) as f:
                query_text = f.read()
            assert query_text is not None
            with open(os.path.join(above10sec, q_idx, 'card_dur.txt'), 'w') as f:
                f.write(str((card, dur)))
        """


        # abo_df = self.formated_generated_evaluated_qs(above10sec, 11, 700)
    def output_new_pp_qs(self, unseen_df, new_qs_folder, test_file):
        base_test_df = pd.read_csv(test_file, sep='\t')
        assert len(set(base_test_df.columns).intersection(list(unseen_df.columns)))
        new_df = pd.concat([base_test_df, unseen_df])
        assert (len(unseen_df)+len(base_test_df)) == len(new_df)
        assert not os.path.exists(new_qs_folder)
        os.makedirs(new_qs_folder)
        new_df.to_csv(os.path.join(new_qs_folder, 'queries.tsv'), sep='\t', index=False)

    def benchmark_slow_qs(self, above10sec, url):
        # Benchmark slow queries
        self.slow_endpoint = Endpoint(url)
        self.slow_endpoint.sparql.setTimeout(900)
        exception_log = os.path.join(self.single_tp_qs, 'queryExecutionExceptions.log')
        lst_qs = os.listdir(above10sec)
        for no, q_idx in enumerate(lst_qs):
            query_path = os.path.join(above10sec, q_idx, 'query_text.txt')
            print(f'[{no}/{len(lst_qs)}]Beginning on {query_path}')
            query_text = None
            with open(query_path) as f:
                query_text = f.read()
            assert query_text is not None
            try:
                start = time.time()
                card = len(self.slow_endpoint.run_query_and_results(query_text))
                dur = time.time() - start
                with open(os.path.join(above10sec, q_idx, 'card_dur.txt'), 'w') as f:
                    f.write(str((card, dur)))
            except TimeoutError:
                card = 0
                dur = 900
                with open(os.path.join(above10sec, q_idx, 'card_dur.txt'), 'w') as f:
                    f.write(str((card, dur)))
            except Exception:
                with open(exception_log, 'a') as f:
                    f.write('---\n')
                    f.write(str(q_idx))
                    f.write(query_text)

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

    def generate_single_tp_queries(self, mid_rel_freqs, outputfolder, url):
        q25 = mid_rel_freqs.quantile(0.25)
        q50 = mid_rel_freqs.quantile(0.5)
        q5 = mid_rel_freqs.quantile(0.05)
        q1 = mid_rel_freqs.quantile(0.01)
        q0 = mid_rel_freqs.quantile(0.)
        q10 = mid_rel_freqs.quantile(0.1)
        # This conf seems generate a lot of medium intervals queries. Let's test with this for now
        gen_qs = self.generate_Simple_qs(q1, q5)
        self.single_tp_qs = os.path.join(outputfolder, "singleTPqs")
        above10sec = os.path.join(self.single_tp_qs, 'above10sec')
        bet_1_10sec = os.path.join(self.single_tp_qs, 'bet1_10sec')
        less1sec = os.path.join(self.single_tp_qs, 'less1sec')
        # create folder structure if not exists
        for i in [self.single_tp_qs, above10sec, bet_1_10sec, less1sec]:
            os.makedirs(i, exist_ok=True)
        no_less1sec = 0
        # utility for above 10 sec qs
        self.endpoint10sec = Endpoint(url)
        self.endpoint10sec.sparql.setTimeout(10)
        no_above10sec = 0
        # utility for between 1 and 10 sec qs
        self.endpoint1sec = Endpoint(url)
        self.endpoint1sec.sparql.setTimeout(1)
        no_bet1_10 = 0
        for idx, q in enumerate(gen_qs):
            print(f"Status: less 1 sec {no_less1sec}, bet 1-10 {no_bet1_10}, above 10 sec: {no_above10sec}")
            try:
                start = time.time()
                card = len(self.endpoint1sec.run_query_and_results(q))
                dur = time.time() - start
                q_fold = os.path.join(less1sec, str(no_less1sec))
                os.makedirs(q_fold)
                with open(os.path.join(q_fold, 'query_text.txt'), 'w') as f:
                    f.write(q)
                with open(os.path.join(q_fold, 'card_dur.txt'), 'w') as f:
                    f.write(str((card, dur)))
                no_less1sec += 1
            except TimeoutError:
                try:
                    start = time.time()
                    card = len(self.endpoint10sec.run_query_and_results(q))
                    dur = time.time() - start
                    q_fold = os.path.join(bet_1_10sec, str(no_bet1_10))
                    os.makedirs(q_fold)
                    with open(os.path.join(q_fold, 'query_text.txt'), 'w') as f:
                        f.write(q)
                    with open(os.path.join(q_fold, 'card_dur.txt'), 'w') as f:
                        f.write(str((card, dur)))
                    no_bet1_10 += 1
                except TimeoutError:
                    q_fold = os.path.join(above10sec, str(no_above10sec))
                    os.makedirs(q_fold)
                    with open(os.path.join(q_fold, 'query_text.txt'), 'w') as f:
                        f.write(q)
                    no_above10sec += 1

    def generate_Simple_qs(self, q25, q50):
        rel_cands = []
        for x in self.pred_freq_unseen.keys():
            freq = self.pred_freq_unseen[x]
            if freq >= q25 and freq <= q50:
                rel_cands.append(x)
        gen_qs = []
        for pred in rel_cands:
            generated_q = f"""SELECT ?o WHERE {{
            ?s <{pred}> ?o .
            }}"""
            gen_qs.append(generated_q)
        return gen_qs

    def query_gen_for_inst_data1(self):
        self.inst_data_folder = os.path.join(self.template1_output, 'inst_data')
        queries = []
        for rel_no in os.listdir(self.inst_data_folder):
            fold_path = os.path.join(self.inst_data_folder, rel_no)
            inst_data = InstanceData1(fold_path)
            gen = inst_data.inst_data_gen()
            no = 0
            for pred1, pred2, pred3 in gen:
                generated_query = f"{self.star_template1(f'<{pred1}>', f'<{pred2}>', f'<{pred3}>')}"
                queryID = f'unseenQuery_instV0_{rel_no}_{no}'
                queries.append({'queryID': queryID, 'queryString': generated_query})
                no += 1
        df = pd.DataFrame(queries)
        path_to_queries = os.path.join(self.inst_data_folder, 'generated_Qs.csv')
        df.to_csv(path_to_queries, index=False)

    def instance_data_generation_1(self, rel_str, sorted_rels):
        print('Beginning instance data generation')
        self.inst_data_folder = os.path.join(self.template1_output, 'inst_data')
        os.makedirs(self.inst_data_folder, exist_ok=True)
        for i, pred in enumerate(sorted_rels):
            print(f'[{i} of {len(sorted_rels)}]')
            try:
                sample_folder = os.path.join(self.inst_data_folder, str(i))
                os.makedirs(sample_folder)
                # pred = sorted_rels[0]
                inst_data_query = f"""
                SELECT DISTINCT ?pred2 ?pred3 WHERE 
                {{  ?s <{pred}> ?o .
                    ?s ?pred2 ?o2 .
                    ?s ?pred3 ?o3 .
                    FILTER ( ?pred2 NOT IN {rel_str} )
                    FILTER ( ?pred3 NOT IN {rel_str} )
                }}
                LIMIT 10
                """
                inst_data = self.endpoint.run_query_and_results(inst_data_query)
                pred_file = os.path.join(sample_folder, 'pred_file.txt')
                with open(pred_file, 'w') as f:
                    f.write(pred)
                query_file = os.path.join(sample_folder, 'inst_query.rq')
                with open(query_file, 'w') as f:
                    f.write(inst_data_query)
                inst_data_file = os.path.join(sample_folder, 'inst_data.pickle')
                with open(inst_data_file, 'wb') as f:
                    pickle.dump(inst_data, f)
            except Exception as e:
                print(e)
                continue

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

    # Use when additional TP has literal values.
    def PP_template1_w_filter(self, pp_pred, pred, ent):
        return f"""
        SELECT ?e ?o2 WHERE {{
        <{ent}> <{pp_pred}>+ ?e .
        ?e <{pred}> ?o2 
        FILTER( LANG(?o) == \"\"en\"\")
        }}
        """


if __name__ == "__main__":
    base_path = '/data/DBpedia_3_class_full'
    train_file = f'{base_path}/train_sampled.tsv'
    val_file = f'{base_path}/val_sampled.tsv'
    test_file = f'{base_path}/test_sampled.tsv'
    url = 'http://172.21.233.14:8891/sparql'
    pred_stat_path = '/data/metaKGStat/dbpedia/predicate/pred_stat/batches_response_stats/freq'
    subj_stat_path = '/data/metaKGStat/dbpedia/entity/ent_stat/batches_response_stats/subj'
    outputfolder = '/data/generatedUnseen'
    new_qs_folder = '/data/DBpedia_3_class_full/newUnseenQs'
    os.makedirs(outputfolder, exist_ok=True)
    UnseenGenerator(train_file, val_file, test_file, pred_stat_path, url, outputfolder, subj_stat_path,new_qs_folder)
