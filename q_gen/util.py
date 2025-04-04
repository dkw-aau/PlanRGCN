import json
import os

from graph_construction.jar_utils import get_query_graph


class Utility:
    @staticmethod
    def get_ent_rels_from_train_val(train_df, val_df):
        train_val_ents = []
        train_val_rels = []
        for idx, row in train_df.iterrows():
            try:
                ents, rels = Utility.get_ent_rel(row['queryString'])
                train_val_rels.extend(rels)
                train_val_ents.extend(ents)
            except Exception:
                continue
        for idx, row in val_df.iterrows():
            try:
                ents, rels = Utility.get_ent_rel(row['queryString'])
                train_val_rels.extend(rels)
                train_val_ents.extend(ents)
            except Exception:
                continue
        return train_val_rels, train_val_ents

    @staticmethod
    def get_ent_rels_from_train(train_df):
        train_ents = []
        train_rels = []
        for idx, row in train_df.iterrows():
            try:
                ents, rels = Utility.get_ent_rel(row['queryString'])
                train_rels.extend(rels)
                train_ents.extend(ents)
            except Exception:
                continue
        return train_rels, train_ents


    @staticmethod
    def get_ent_rel(query):
        qg = get_query_graph(query)
        rels = set()
        ents = set()
        for n in qg['nodes']:
            if n['nodeType'] == 'PP' or n['nodeType'] == 'TP':
                if 'http' in n['subject']:
                    ents.add(n['subject'])
                if 'http' in n['object']:
                    ents.add(n['object'])
                if n['nodeType'] == 'PP':
                    for p in n['predicateList']:
                        if 'http' in p:
                            if p.startswith('<'):
                                p = p[1:]
                            if p.endswith('>'):
                                p = p[:-1]
                            rels.add(p)
                else:  # will always be TP
                    p = n['predicate']
                    if 'http' in p:
                        if p.startswith('<'):
                            p = p[1:]
                        if p.endswith('>'):
                            p = p[:-1]
                        rels.add(p)
        return ents, rels

    @staticmethod
    def get_pred_freq(pred_stat_path):
        pred_freq = {}
        pred_freq_paths = [os.path.join(pred_stat_path, x) for x in os.listdir(pred_stat_path)]
        for p in pred_freq_paths:
            data = json.load(open(p, 'r'))
            for s in data['results']['bindings']:
                pred_freq[s['p1']['value']] = int(s['triples']['value'])
        return pred_freq

    @staticmethod
    def get_subj_freq(subj_stat_path):
        pred_freq = {}
        subj_stat_paths = [os.path.join(subj_stat_path, x) for x in os.listdir(subj_stat_path)]
        for p in subj_stat_paths:
            data = json.load(open(p, 'r'))
            for s in data['results']['bindings']:
                pred_freq[s['e']['value']] = int(s['entities']['value'])
        return pred_freq