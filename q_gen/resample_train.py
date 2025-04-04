import sys


sys.path.append('/PlanRGCN/')
import os
os.environ['QG_JAR'] = '/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar'
os.environ['QPP_JAR'] = '/PlanRGCN/qpp/qpp_features/sparql-query2vec/target/sparql-query2vec-0.0.1.jar'
import time
from q_gen.util import Utility

import json

import pickle

import pandas as pd


from feature_extraction.sparql import Endpoint
from graph_construction.jar_utils import get_ent_rel
class Resampler:
    def __init__(self, train_sampled, val_sampled, test_sampled):
        self.train_sampled = train_sampled
        self.val_sampled = val_sampled
        self.test_sampled = test_sampled

        train_df = pd.read_csv(train_sampled, sep='\t')
        val_df = pd.read_csv(val_sampled, sep='\t')
        test_df = pd.read_csv(test_sampled, sep='\t')

        train_ents = []
        train_rels = []

        # The train map should contain queries that are in the
        rdf_train_map = {}
        # RDF term in slow and medium interval that should be removed from the train map
        rdf_term_s_m = []
        for idx, row in train_df.iterrows():
            try:
                ents, rels = Utility.get_ent_rel(row['queryString'])
                #extract rdf term in slow and medium queries
                if row['mean_latency'] > 1:
                    rdf_term_s_m.extend(ents)
                    rdf_term_s_m.extend(rels)

                # Create rdf2trainqueries map
                for lst in [ents, rels]:
                    for r in lst:
                        try:
                            rdf_train_map[r].append(row['queryID'])
                        except KeyError:
                            rdf_train_map[r] = [row['queryID']]
            except Exception:
                continue
        """
        # Remove rdf terms that are part of slow or medium queries in train
        for term in rdf_term_s_m:
            try:
                del rdf_train_map[term]
            except KeyError:
                pass
        """

        q2rdf_test = {}
        #create medium query to rdf term
        for idx, row in test_df.iterrows():
            try:
                ents, rels = Utility.get_ent_rel(row['queryString'])
                #extract rdf term in slow and medium queries

                if 1 < row['mean_latency'] < 10:
                    # Create rdf2test map
                    for lst in [ents, rels]:
                        for r in lst:

                            try:
                                q2rdf_test[row['queryID']].append(r)
                            except KeyError:
                                q2rdf_test[row['queryID']] = [r]
            except Exception:
                continue


        #return number of train queires to remvoe
        def train_query_idier(test_query):
            global q2rdf_test
            global rdf_train_map
            rdfs = q2rdf_test[test_query]
            card = 0
            for rdf in rdfs:
                if rdf in rdf_train_map.keys():
                    card += len(rdf_train_map[rdf])
            return card

        test_qs_sorted = [x for x in q2rdf_test.keys()]
        test_qs_sorted = sorted(test_qs_sorted, key=train_query_idier)


        testqs2train = {}
        for test_query_id in q2rdf_test.keys():
            rdfs = q2rdf_test[test_query_id]
            testqs2train[test_query_id] = []
            for term in rdfs:
                try:
                    testqs2train[test_query_id].extend(rdf_train_map[term])
                except KeyError:
                    pass
        testqs_sorted = sorted([x for x in testqs2train.keys()], key= lambda x: testqs2train[x])

        no_unseen_med = 14
        #Selected train query ids and rdf terms to make completely unseen
        selected_queryIds = []
        unseen_test_queries = []
        cur_rdf_idx = 0
        while len(unseen_test_queries) < no_unseen_med:
            test_query_id = testqs_sorted[cur_rdf_idx]
            selected_queryIds.extend(testqs2train[test_query_id])
            unseen_test_queries.append(test_query_id)
            cur_rdf_idx += 1

        new_train


base = '/data/unseen_DBpedia_3_class_full/before_drop'

train_sampled = os.path.join(base,'train_sampled.tsv')
val_sampled = os.path.join(base, 'val_sampled.tsv')
test_sampled = os.path.join(base,'test_sampled.tsv')
Resampler(train_sampled=train_sampled, val_sampled=val_sampled, test_sampled=test_sampled)
