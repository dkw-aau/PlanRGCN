import numpy as np
import numpy as np
from feature_extraction.entity_features import EntityFeatures
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery

from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj

class Node:
    pred_bins=30
    pred_topk=15
    pred_feaurizer: PredicateFeaturesQuery=None
    pred_feat_sub_obj_no: bool=None
    use_ent_feat:bool=False
    ent_bins:int = None
    use_join_features:bool=True
    ent_featurizer:EntityFeatures = None
    
    def __init__(self, node_label:str) -> None:
        self.node_label = node_label
        if node_label.startswith('?'):
            self.type = 'VAR'
        elif node_label.startswith('<http') or node_label.startswith('http'):
            self.type = 'URI'
        elif node_label.startswith('join'):
            self.type = 'JOIN'
        else:
            self.type = None
        
        self.pred_freq = None
        self.pred_literals = None
        self.pred_entities= None
        #self.topK = None
        
        #for join node
        self.is_subject_var =None
        self.is_pred_var =None
        self.is_object_var=None
    
    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f'{self.type} {self.node_label}'
    def __eq__(self, other):
        return self.node_label == other.node_label
    def __hash__(self) -> int:
        return hash(self.node_label)
    
    def get_pred_features(self):
        pred_bins, pred_topk,pred_feat_sub_obj_no= Node.pred_bins, Node.pred_topk,Node.pred_feaurizer
        
        predicate_bins = np.zeros(pred_bins)
        topk_vec = np.zeros(pred_topk)
        
        if pred_feat_sub_obj_no:
            predicate_features = np.zeros(4)
        else:
            predicate_features = np.zeros(3)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            predicate_features[1] = self.pred_literals
            if not pred_feat_sub_obj_no:
                predicate_features[2] = self.pred_entities
            else:
                predicate_features[2] = self.pred_subject_count
                predicate_features[3] = self.pred_object_count
            try:
                predicate_bins[self.bucket] = 1
            except AttributeError:
                predicate_bins[pred_bins-1] = 1
            if self.topK != None and self.topK < pred_topk:
                topk_vec[self.topK] = 1
        if np.sum(np.isnan( predicate_features)) > 0:
            predicate_features[np.isnan(predicate_features)] = 0
            #raise Exception
        if np.sum(np.isnan( predicate_bins)) > 0:
            raise Exception
        if np.sum(np.isnan( topk_vec)) > 0:
            raise Exception
        return np.concatenate(( predicate_features, predicate_bins, topk_vec))
        #return predicate_features,predicate_bins, topk_vec
    
    def get_ent_features(self, ent_bins):
        freq_vec_ent = np.zeros(1)
        ent_bins_vec = np.zeros(ent_bins+1)
        if self.nodetype in [0,2] and self.type == 'URI':
            freq_vec_ent[0] = self.ent_freq
            ent_bins_vec[self.ent_bin] = 1
        if np.sum(np.isnan( freq_vec_ent)) > 0:
            raise Exception
        if np.sum(np.isnan( ent_bins_vec)) > 0:
            raise Exception
        return np.concatenate((freq_vec_ent,ent_bins_vec))
    
    def get_join_features(self):
        join_feat = np.zeros(3)
        if self.nodetype == 3:
            return join_feat
        if self.is_subject_var:
            join_feat[0] =1
        if self.is_pred_var:
            join_feat[1] =1
        if self.is_object_var:
            join_feat[2] =1
        return join_feat
    
    def get_features(self):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = self.get_pred_features()
        if Node.use_ent_feat:
            ent_features = self.get_ent_features(Node.ent_bins)
        else:
            ent_features = np.array([])
        if Node.use_join_features:
            join_feat = self.get_join_features()
        else:
            join_feat = np.array([])
        return np.concatenate((nodetype ,join_feat, predicate_features, ent_features))
    
    def set_predicate_features(self):
        self.pred_freq = -1
        self.pred_literals = -1
        self.pred_subject_count,self.pred_object_count =-1,-1
        predicate_stat = Node.pred_feaurizer
        
        if predicate_stat != None:
            #self.predicate_stat = predicate_stat
            if self.type == 'URI':
                self.bucket = int(predicate_stat.get_bin(self.node_label))
                self.topK = predicate_stat.top_k_predicate(self.node_label)
                if self.bucket == None:
                    self.bucket = 0
                
                if self.node_label in predicate_stat.predicate_freq.keys():
                    self.pred_freq = predicate_stat.predicate_freq[self.node_label]
                else:
                    self.pred_freq = -1    
                
                if self.node_label in predicate_stat.uniqueLiteralCounter.keys():
                    self.pred_literals = predicate_stat.uniqueLiteralCounter[self.node_label]
                else:
                    self.pred_literals = -1  
                
                if (not isinstance(predicate_stat,Predicate_Featurizer_Sub_Obj)) and (self.node_label in predicate_stat.unique_entities_counter.keys()):
                    self.pred_entities = predicate_stat.unique_entities_counter[self.node_label]
                else:
                    self.pred_entities = -1
                
                if (isinstance(predicate_stat,Predicate_Featurizer_Sub_Obj)) and (self.node_label in predicate_stat.unique_entities_counter.keys()):
                    self.pred_subject_count,self.pred_object_count = predicate_stat.unique_entities_counter[self.node_label]
                else:
                    self.pred_subject_count = -1
                    self.pred_object_count = -1
    
    def set_entity_feature(self):
        ent_featurizer = Node.ent_featurizer
        if ent_featurizer == None:
            return self
        if self.type == 'URI':
            bin_no,freq = ent_featurizer.get_feature(self.node_label)
            self.ent_bin = bin_no
            self.ent_freq = freq
            return self