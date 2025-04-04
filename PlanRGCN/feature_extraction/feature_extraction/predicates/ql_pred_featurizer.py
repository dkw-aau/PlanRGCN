

import configparser
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
import pandas as pd

from graph_construction.bgp import BGP
from graph_construction.nodes.node import Node
from graph_construction.triple_pattern import TriplePattern
from preprocessing.utils import get_predicates_from_path, load_BGPS_from_json
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

class ql_pred_featurizer(Predicate_Featurizer_Sub_Obj):
    
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)
    
    #this function will be invoked during the loading of the function. Adding only to override existing function.
    def prepare_pred_feat(self, bins = 30, k=20):
        return
    
    def prepare_featurizer(self,preds : list[str], k, buckets):
        pred_mapper = {}
        for pred in preds:
            if pred in pred_mapper.keys():
                pred_mapper[pred] += 1
            else:
                pred_mapper[pred] = 1
        """for bgp in bgps:
            for triple in bgp.triples:
                triple:TriplePattern
                if triple.predicate.node_label in pred_mapper.keys():
                    pred_mapper[triple.predicate.node_label] += 1
                else:
                    pred_mapper[triple.predicate.node_label] = 1"""
        self.pred_mapper = pred_mapper
        print(f"Prediate map initialised with {len(list(pred_mapper.keys()))} predicates")
        self.add_predicate_binner_topK(k, buckets)
    
    #prepares and adds topk and buckets for 
    def add_predicate_binner_topK(self, k, buckets):    
        dct = {'predicate':[], 'freq':[]}
        for key in self.pred_mapper.keys():
            dct['predicate'].append(key)
            dct['freq'].append(self.pred_mapper[key])
        df = pd.DataFrame.from_dict(dct)
        #df_freq = df['freq'].astype('int')
        df['freq'] = pd.to_numeric(df['freq'])
        df_freq = df.sort_values('freq',ascending=False)
        df = df_freq
        df_freq = df_freq.assign(row_number=range(len(df_freq)))
        df_freq = df_freq.set_index('row_number')
        
        #print(df_freq.columns)
        #print(df_freq.index)
        df_freq = df_freq.loc[:k]
        df_freq = df_freq.reset_index().set_index('predicate')
        
        #equal width binning
        self.est = KBinsDiscretizer(n_bins=buckets, encode='ordinal', 
                       strategy='uniform')
        df['bin'] = self.est.fit_transform(df[['freq']])
        df['bin'] = df['bin'].apply(lambda x: int(x))
        #quantile binning
        #df['bin'], cut_bin = pd.qcut(df['freq'], q = buckets, labels = [x for x in range(buckets)], retbins = True)
        #print(cut_bin)
        #max_bin = df['bin'].max()
        df = df.set_index('predicate')
        self.predicate_bin_df = df
        #self.bin_vals = cut_bin
        self.freq_k = k+1
        #self.total_bin = len(cut_bin)+1
        self.total_bin = np.max(self.predicate_bin_df['bin'])+1
        self.topk_df = df_freq
    def top_k_predicate(self, predicate):
        if not hasattr(self,'topk_df'):
            raise RuntimeError("prepare_featurizer() need to be called first!!")
        try:
            return self.topk_df.loc[predicate]['row_number']
        except KeyError:
            return None
    
    def get_bin(self, predicate):
        if not hasattr(self,'predicate_bin_df'):
            raise RuntimeError("prepare_featurizer() need to be called first!!")
        try:
            return self.predicate_bin_df.loc[predicate]['bin']
        except KeyError as e:
            #print(predicate)
            return self.total_bin

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    data_file = parser['Dataset']['train_data_path']
    train_bgps = get_predicates_from_path(data_file)
    feat = ql_pred_featurizer()
    feat.prepare_featurizer(train_bgps, 30,40)
    
    
        