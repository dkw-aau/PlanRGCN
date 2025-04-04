import networkx as nx
import pandas as pd
import pickle
from graph_construction.jar_utils import get_query_graph
from collections import Counter
import numpy as np
def flatten_extend(matrix):
     flat_list = []
     for row in matrix:
         flat_list.extend(row)
     return flat_list

class UnseenSplitter:
    def __init__(self, train_file, val_file, test_file, datasplit_path, new_train_file, new_val_file, new_test_file, selected_train_ids):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.n_train_file = new_train_file
        self.n_val_file = new_val_file
        self.n_test_file = new_test_file

        self.train_df = pd.read_csv(train_file, sep ='\t')
        self.val_df = pd.read_csv(val_file, sep='\t')
        self.test_df = pd.read_csv(test_file, sep='\t')
        self.datasplit_path = datasplit_path
        self.datasplitter = pickle.load(open(self.datasplit_path, 'rb'))
        self.interval_set = set(self.datasplitter.interval_extractor.intervals)
        self.min_completelyUnseen = 100
        #The number at training queries that should at most be removed per test query
        self.remove_at_most_tq = 30 #20

        def invert_dct(dct:dict):
            n_dct = {}
            for k in dct:
                for v in dct[k]:
                    try:
                        n_dct[v].append(k)
                    except KeyError:
                        n_dct[v] = [k]
            return n_dct

        #Mapping of rdf terms in train and val to queryID
        rdf2q_non_test = self.get_ent_rels_df(self.train_df)
        rdf2q_non_test = self.get_ent_rels_df(self.val_df, dct=rdf2q_non_test)
        # Mapping of rdf terms in test to queryID
        rdf2q_test = self.get_ent_rels_df(self.test_df)

        #inverted maps
        q2rdf_non_test = invert_dct(rdf2q_non_test)
        q2rdf_test = invert_dct(rdf2q_test)

        # RDF terms in both test and one of train of val sets
        testRDFinTrainVal = set(rdf2q_test.keys()).intersection(set(rdf2q_non_test.keys()))

        # RDF terms in test but not in train or val sets
        testRDFNotInTrainVal = set([x for x in rdf2q_test.keys() if x not in testRDFinTrainVal])

        test_ids_RDF_not_in_train = flatten_extend([rdf2q_test[x] for x in testRDFNotInTrainVal])



        test_ids_RDF_in_train = flatten_extend([rdf2q_test[x] for x in testRDFinTrainVal])
        old_rt_dist_unseen =  self.unseen_distribution(self.train_df, self.val_df, self.test_df)

        #Compute currently missing interval
        partially_unseen_ids = []
        completely_unseen_ids = []
        for test_id in test_ids_RDF_not_in_train:
            test_id_RDF = [x for x in rdf2q_test.keys() if test_id in rdf2q_test[x]]
            if len(set(test_id_RDF).intersection(testRDFinTrainVal)) > 0:
                partially_unseen_ids.append(test_id)
            else:
                completely_unseen_ids.append(test_id)
        self.missing_intervals = self.get_rt_missing(completely_unseen_ids, self.test_df)
        exit() # temp code
        #Test completel unssen queries
        completety_unseen_df = self.test_df.loc[self.test_df['queryID'].isin(completely_unseen_ids)].copy()
        completety_unseen_df['interval'] = completety_unseen_df['mean_latency'].apply(lambda x: self.interval_check(x))
        completely_unseen_dist = dict(Counter(completety_unseen_df['interval']))
        rdf_terms_completely_unseen = list(self.get_ent_rels_df(completety_unseen_df).keys())

        # Test queries with RDF term in train/val and their runtime intervals
        test_RDF_train_df = self.test_df.loc[self.test_df['queryID'].isin(test_ids_RDF_in_train)].copy()
        test_RDF_train_df['interval'] = test_RDF_train_df['mean_latency'].apply(lambda x: self.interval_check(x))

        #frequency map of rdf terms to train queries
        frequentTrainTerms = {}
        for term in rdf2q_non_test.keys():
            frequentTrainTerms[term] = len(rdf2q_non_test[term])

        #Given a missing interval, we want to identifity a set of test queries where we can remove certain train queries to make them unseen
        test_queries_unseen = []  # test queries becoming unseen
        train_queries_rm = []  # training queries to remove to make test queries unseen
        for m in self.missing_intervals:
            test_mis_interval = test_RDF_train_df.loc[test_RDF_train_df['interval'] == m]


            for idx, row in test_mis_interval.iterrows():
                t_query_terms = q2rdf_test[row['queryID']]
                affected_train_queries = np.sum([frequentTrainTerms[t] for t in t_query_terms if t in testRDFinTrainVal])
                if affected_train_queries > self.remove_at_most_tq:
                    continue
                for t in t_query_terms:
                    if t in testRDFinTrainVal:
                        train_queries_rm.extend(rdf2q_non_test[t])
                    rdf_terms_completely_unseen.append(t)
                test_queries_unseen.append(row['queryID'])
            train_queries_rm = list(set(train_queries_rm))

        # given the list of train queries to remove and the set of new unseen quiries, we extract train queries transferable to test set
        transfer_train_df = pd.concat([self.train_df.loc[self.train_df['queryID'].isin(train_queries_rm)].copy(),self.val_df.loc[self.val_df['queryID'].isin(train_queries_rm)].copy()], ignore_index=True)
        assert len(train_queries_rm) == len(transfer_train_df)
        transfer_train_df['interval'] = transfer_train_df['mean_latency'].apply(lambda x: self.interval_check(x))
        transfer_filt = transfer_train_df['queryString'].apply(lambda x: len(self.get_rdf_terms(x).intersection(rdf_terms_completely_unseen)) == len(self.get_rdf_terms(x)))
        transfer_qs_df = transfer_train_df[transfer_filt].copy()

        # remove queries from train and val for unseen test queries
        n_train = self.train_df.loc[~self.train_df['queryID'].isin(train_queries_rm)].copy()
        n_val = self.val_df.loc[~self.val_df['queryID'].isin(train_queries_rm)].copy()
        n_test = pd.concat([self.test_df, transfer_qs_df], ignore_index=True)

        # Compute new completely unseen queries
        new_rt_dist = self.unseen_distribution(n_train, n_val, n_test)

        n_train.to_csv(self.n_train_file, sep='\t', index=False)
        n_val.to_csv(self.n_val_file, sep='\t', index=False)
        n_test.to_csv(self.n_test_file, sep='\t', index=False)

    def interval_check(self, rt):
        for interval in self.interval_set:
            if interval[0]< rt<= interval[1]:
                return interval
        return None

    def get_rt_missing(self, ids, full_df):
        intvals = []
        counts = self.get_rt_dist(ids, full_df)
        for k in counts.keys():
            if counts[k]< self.min_completelyUnseen:
                intvals.append(k)
        return intvals

    def get_missing_number(self, ids, full_df, interval):
        counts = self.get_rt_dist(ids, full_df)
        return counts[interval]

    def get_rt_dist(self, ids, full_df:pd.DataFrame):
        df = full_df[full_df['queryID'].isin(ids)].copy()
        df['interval'] = df['mean_latency'].apply(lambda x: self.interval_check(x))
        counts = dict(Counter(df['interval']))
        return counts

    def get_interval_from_id(self, id):
        df = self.train_df[self.train_df.queryID == id]
        if len(df) == 0:
            df =self.val_df[self.val_df.queryID == id]
        return self.interval_check(df['mean_latency'].iloc[0])

    def get_ent_rels_df(self, df, dct = None):
        normalize = lambda term: term.replace('<', '').replace('>', '')


        RDF2q = {} if dct is None else dct
        for idx, row in df.iterrows():
            qg = get_query_graph(row['queryString'])
            for n in qg['nodes']:
                match n['nodeType']:
                    case "PP":
                        pred = normalize(n['predicateList'][0])
                        try:
                            RDF2q[pred].append(row['queryID'])
                        except KeyError:
                            RDF2q[pred] = [row['queryID']]
                    case "TP":
                        if n['predicate'].startswith('http') or n['predicate'].startswith('<http'):
                            pred = normalize(n['predicate'])
                            try:
                                RDF2q[pred].append(row['queryID'])
                            except KeyError:
                                RDF2q[pred] = [row['queryID']]
                    case "FILTER":
                        continue
                    case _:
                        ...

                obj = n['object']
                if obj.startswith('http') or obj.startswith('<http'):
                    try:
                        RDF2q[normalize(obj)].append(row['queryID'])
                    except KeyError:
                        RDF2q[normalize(obj)] = [row['queryID']]
                subj = n['subject']
                if subj.startswith('http') or subj.startswith('<http'):
                    try:
                        RDF2q[normalize(subj)].append(row['queryID'])
                    except KeyError:
                        RDF2q[normalize(subj)] = [row['queryID']]
        return RDF2q

    def get_rdf_terms(self, query):
        normalize = lambda term: term.replace('<', '').replace('>', '')
        qg = get_query_graph(query)
        terms = set()
        for n in qg['nodes']:
            match n['nodeType']:
                case "PP":
                    pred = normalize(n['predicateList'][0])
                    terms.add(pred)
                case "TP":
                    if n['predicate'].startswith('http') or n['predicate'].startswith('<http'):
                        pred = normalize(n['predicate'])
                        terms.add(pred)
                case "FILTER":
                    continue
                case _:
                    ...

            obj = n['object']
            if obj.startswith('http') or obj.startswith('<http'):
                terms.add(normalize(obj))
            subj = n['subject']
            if subj.startswith('http') or subj.startswith('<http'):
                terms.add(normalize(subj))
        return terms
    def unseen_distribution(self, train_df, val_df, test_df):
        rdf2q_non_test = {}
        rdf2q_non_test = self.get_ent_rels_df(train_df, dct = rdf2q_non_test)
        rdf2q_non_test = self.get_ent_rels_df(val_df, dct=rdf2q_non_test)
        rdf2q_test = self.get_ent_rels_df(test_df)

        testRDFinTrainVal = set(rdf2q_test.keys()).intersection(set(rdf2q_non_test.keys()))
        testRDFNotInTrainVal = set([x for x in rdf2q_test.keys() if x not in testRDFinTrainVal])
        test_ids_RDF_not_in_train = flatten_extend([rdf2q_test[x] for x in testRDFNotInTrainVal])

        partially_unseen_ids = []
        completely_unseen_ids = []
        for test_id in test_ids_RDF_not_in_train:
            test_id_RDF = [x for x in rdf2q_test.keys() if test_id in rdf2q_test[x]]
            if len(set(test_id_RDF).intersection(testRDFinTrainVal)) > 0:
                partially_unseen_ids.append(test_id)
            else:
                completely_unseen_ids.append(test_id)

        return self.get_rt_dist(completely_unseen_ids, test_df)

